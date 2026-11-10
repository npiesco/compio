//! Assert that a funded spawn allocates exactly [`task_layout`], refuses
//! without allocating, and drops its funding only after the task's block is
//! freed, measured with a counting allocator.
//!
//! With the `console` feature, a funded task is not reported, as its span would
//! be allocated by the subscriber: these hold with a subscriber installed too.
#![cfg(not(loom))]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    future::Future,
    pin::pin,
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Waker},
};

use compio_executor::{Executor, ExecutorConfig, task_layout};

/// Counts the bytes live on each thread, and refuses every allocation on a
/// thread while it is told to.
struct Counting;

#[global_allocator]
static ALLOCATOR: Counting = Counting;

std::thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static REFUSE: Cell<bool> = const { Cell::new(false) };
    /// The bytes live when a [`Funding`] was last dropped.
    static DROPPED_AT: Cell<Option<isize>> = const { Cell::new(None) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if REFUSE.try_with(Cell::get).unwrap_or(false) {
            return ptr::null_mut();
        }
        let block = unsafe { System.alloc(layout) };
        if !block.is_null() {
            let _ = LIVE.try_with(|live| live.set(live.get() + layout.size() as isize));
        }
        block
    }

    unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
        unsafe { System.dealloc(block, layout) };
        let _ = LIVE.try_with(|live| live.set(live.get() - layout.size() as isize));
    }
}

fn live() -> isize {
    LIVE.get()
}

/// Pays for a task: dropped, it records the bytes live by then.
struct Funding;

impl Drop for Funding {
    fn drop(&mut self) {
        DROPPED_AT.set(Some(live()));
    }
}

fn block_on<F: Future>(exe: &Executor, f: F) -> F::Output {
    let cx = &mut Context::from_waker(Waker::noop());
    let mut f = pin!(f);
    loop {
        if let Poll::Ready(res) = f.as_mut().poll(cx) {
            return res;
        }
        exe.tick();
    }
}

fn layout_of<F: Future + 'static, C: 'static>(_: &F) -> Layout {
    task_layout::<F, C>()
}

/// Run a funded task to its end, leaving room in the queue for the next.
fn warm(exe: &Executor) {
    let Ok(handle) = exe.try_spawn_funded(async {}, Funding) else {
        panic!("the warming spawn was refused");
    };
    block_on(exe, handle).unwrap();
}

#[test]
fn a_funded_spawn_allocates_its_task_layout_and_drops_the_funding_after_the_block() {
    let exe = Executor::new();
    warm(&exe);

    let task = async { 7usize };
    let layout = layout_of::<_, Funding>(&task);
    DROPPED_AT.set(None);
    let before = live();
    let Ok(handle) = exe.try_spawn_funded(task, Funding) else {
        panic!("the spawn was refused");
    };
    assert_eq!(
        live() - before,
        layout.size() as isize,
        "the spawn allocates its task layout, and nothing else"
    );
    assert_eq!(DROPPED_AT.get(), None, "the funding is held with the task");

    assert_eq!(block_on(&exe, handle).unwrap(), 7);
    assert_eq!(
        DROPPED_AT.get(),
        Some(before),
        "the funding is dropped only after the task's block is freed"
    );
}

#[test]
fn a_funded_task_dropped_unfinished_drops_the_funding_after_the_block() {
    let exe = Executor::new();
    warm(&exe);

    DROPPED_AT.set(None);
    let before = live();
    let Ok(handle) = exe.try_spawn_funded(std::future::pending::<()>(), Funding) else {
        panic!("the spawn was refused");
    };
    exe.tick();
    drop(handle);
    assert_eq!(DROPPED_AT.get(), None, "the queue still holds the task");

    exe.clear();
    assert_eq!(
        DROPPED_AT.get(),
        Some(before),
        "the funding is dropped only after the task's block is freed"
    );
}

/// Spawn, refusing every allocation, and check that nothing was allocated, the
/// funding was dropped, and the future was handed back.
fn refused_spawn(exe: &Executor) {
    DROPPED_AT.set(None);
    let before = live();
    REFUSE.set(true);
    let refused = exe.try_spawn_funded(async { 7usize }, Funding);
    REFUSE.set(false);
    assert_eq!(live(), before, "a refused spawn allocates nothing");
    assert_eq!(
        DROPPED_AT.get(),
        Some(before),
        "a refused spawn drops its funding"
    );
    let Err(task) = refused else {
        panic!("the spawn was granted");
    };
    assert_eq!(block_on(exe, task), 7, "the future is handed back whole");
}

#[test]
fn a_spawn_refused_room_in_the_queue_allocates_nothing() {
    let exe = Executor::with_config(ExecutorConfig {
        local_queue_size: 0,
        ..ExecutorConfig::default()
    });
    refused_spawn(&exe);
}

#[test]
fn a_spawn_refused_its_block_allocates_nothing() {
    let exe = Executor::new();
    warm(&exe);
    refused_spawn(&exe);
}

#[test]
fn a_funded_task_released_on_another_thread_is_freed_and_unfunded_on_its_own() {
    let exe = Executor::new();
    warm(&exe);

    DROPPED_AT.set(None);
    // Cancelled and removed: its parked waker is the last reference.
    let (waker, layout) = parked_alone(&exe, Funding);
    std::thread::spawn(move || drop(waker)).join().unwrap();
    assert_eq!(
        DROPPED_AT.get(),
        None,
        "the funding is not dropped on another thread, nor leaked"
    );

    let held = live();
    exe.tick();
    assert_eq!(
        held - live(),
        layout.size() as isize,
        "the block is freed on the executor's thread"
    );
    assert_eq!(
        DROPPED_AT.get(),
        Some(live()),
        "and the funding dropped there, only after it"
    );
}

/// Spawns a task funded by `funding` that parks its waker, and cancels it,
/// leaving that waker as the last reference to its block: returns the waker and
/// the layout of the block.
fn parked_alone<C: 'static>(exe: &Executor, funding: C) -> (Waker, Layout) {
    let parked = Arc::new(Mutex::new(None::<Waker>));
    let task = {
        let parked = parked.clone();
        std::future::poll_fn(move |cx| {
            *parked.lock().unwrap() = Some(cx.waker().clone());
            Poll::<()>::Pending
        })
    };
    let layout = layout_of::<_, C>(&task);
    let Ok(handle) = exe.try_spawn_funded(task, funding) else {
        panic!("the spawn was refused");
    };
    exe.tick();
    drop(handle);
    exe.tick();
    let waker = parked
        .lock()
        .unwrap()
        .take()
        .expect("the task parked its waker");
    (waker, layout)
}

#[test]
fn a_funded_task_released_on_another_thread_after_its_executor_is_dropped_is_freed_on_its_own() {
    let exe = Executor::new();
    let next = Executor::new();
    warm(&exe);
    warm(&next);

    DROPPED_AT.set(None);
    let (waker, layout) = parked_alone(&exe, Funding);
    drop(exe);
    assert_eq!(DROPPED_AT.get(), None, "the waker still holds the block");

    std::thread::spawn(move || drop(waker)).join().unwrap();
    assert_eq!(
        DROPPED_AT.get(),
        None,
        "the funding is not dropped on another thread, nor leaked"
    );

    let held = live();
    next.tick();
    assert_eq!(
        DROPPED_AT.get(),
        Some(held - layout.size() as isize),
        "the block is freed on the dropped executor's thread as another there ticks, and the \
         funding dropped there only after it"
    );
}

/// The bytes live on its thread as a [`Homed`] was dropped.
static HOMED_DROPPED_AT: Mutex<Option<isize>> = Mutex::new(None);

/// Pays for a task: dropped, it records the bytes live on its thread by then.
struct Homed;

impl Drop for Homed {
    fn drop(&mut self) {
        *HOMED_DROPPED_AT.lock().unwrap() = LIVE.try_with(Cell::get).ok();
    }
}

#[test]
fn a_funded_task_released_on_another_thread_after_its_executor_is_dropped_is_freed_as_its_thread_exits()
 {
    let (send, parked) = std::sync::mpsc::channel();
    let (exit, exiting) = std::sync::mpsc::channel::<()>();
    let home = std::thread::spawn(move || {
        let exe = Executor::new();
        let (waker, layout) = parked_alone(&exe, Homed);
        drop(exe);
        send.send(waker).unwrap();
        exiting.recv().unwrap();
        (live(), layout)
    });
    let waker = parked.recv().unwrap();
    let before = live();
    drop(waker);
    assert_eq!(live(), before, "the block is not freed on another thread");
    assert_eq!(
        *HOMED_DROPPED_AT.lock().unwrap(),
        None,
        "the funding is not dropped on another thread"
    );

    exit.send(()).unwrap();
    let (exiting, layout) = home.join().unwrap();
    let dropped_at = HOMED_DROPPED_AT
        .lock()
        .unwrap()
        .expect("the funding is dropped as its thread exits");
    assert!(
        dropped_at <= exiting - layout.size() as isize,
        "the block is freed on its thread as it exits, before the funding is dropped"
    );
}

/// Whether a [`Stranded`] was dropped.
static STRANDED_DROPPED: AtomicBool = AtomicBool::new(false);

/// Pays for a task: dropped, it records that it was.
struct Stranded;

impl Drop for Stranded {
    fn drop(&mut self) {
        STRANDED_DROPPED.store(true, Ordering::SeqCst);
    }
}

#[test]
fn a_funded_task_released_after_its_thread_exited_is_freed_where_it_goes_its_funding_kept() {
    let (waker, layout) = std::thread::spawn(|| parked_alone(&Executor::new(), Stranded))
        .join()
        .unwrap();
    let before = live();
    drop(waker);
    assert_eq!(
        before - live(),
        layout.size() as isize,
        "the block is freed where its last reference goes"
    );
    assert!(
        !STRANDED_DROPPED.load(Ordering::SeqCst),
        "the funding is not dropped on a thread not its own"
    );
}

#[cfg(feature = "console")]
#[test]
fn a_funded_spawn_allocates_no_console_span() {
    let _subscriber = tracing::subscriber::set_default(tracing_subscriber::registry());
    let exe = Executor::new();
    warm(&exe);

    let task = async {};
    let layout = layout_of::<_, Funding>(&task);
    let before = live();
    let Ok(handle) = exe.try_spawn_funded(task, Funding) else {
        panic!("the spawn was refused");
    };
    assert_eq!(
        live() - before,
        layout.size() as isize,
        "nothing but the block, no span"
    );
    block_on(&exe, handle).unwrap();

    // The first span the subscriber holds allocates its storage, which the
    // spawn above would have shown had it made one.
    let before = live();
    let span = tracing::trace_span!("runtime.spawn");
    assert!(live() > before, "the subscriber allocates a span");
    drop(span);
}
