//! Assert that a funded spawn allocates exactly [`task_layout`], refuses
//! without allocating, and drops its funding only after the task's block is
//! freed, measured with a counting allocator.
//!
//! The console's spans may allocate as a task is spawned, so these run without
//! it.
#![cfg(not(any(feature = "console", loom)))]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    future::Future,
    pin::pin,
    ptr,
    sync::{Arc, Mutex},
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

    let parked = Arc::new(Mutex::new(None::<Waker>));
    let task = {
        let parked = parked.clone();
        std::future::poll_fn(move |cx| {
            *parked.lock().unwrap() = Some(cx.waker().clone());
            Poll::<()>::Pending
        })
    };
    let layout = layout_of::<_, Funding>(&task);
    DROPPED_AT.set(None);
    let Ok(handle) = exe.try_spawn_funded(task, Funding) else {
        panic!("the spawn was refused");
    };
    exe.tick();
    // Cancelled and removed: its parked waker is the last reference.
    drop(handle);
    exe.tick();
    let waker = parked
        .lock()
        .unwrap()
        .take()
        .expect("the task parked its waker");
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
