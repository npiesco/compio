//! Assert that a funded operation refuses without allocating, and keeps its
//! funding until the block it is kept in is freed: when its result is taken,
//! when a cancel token outlives it, or when the kernel completes it after its
//! key was dropped, measured with a counting allocator.
#![cfg(unix)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    os::fd::OwnedFd,
    ptr,
};

use compio_buf::{BufResult, IntoInner};
use compio_driver::{
    Key, OpCode, Proactor, PushEntry,
    op::{Pipe, Read, Write},
};

/// Counts the bytes live on each thread, refuses every allocation on a thread
/// while it is told to, and notes when the watched block is freed.
struct Counting;

#[global_allocator]
static ALLOCATOR: Counting = Counting;

std::thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static REFUSE: Cell<bool> = const { Cell::new(false) };
    static WATCHED: Cell<usize> = const { Cell::new(0) };
    static WATCHED_FREED: Cell<bool> = const { Cell::new(false) };
    /// Whether the watched block was freed when a [`Funding`] was last
    /// dropped.
    static DROPPED: Cell<Option<bool>> = const { Cell::new(None) };
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
        if WATCHED
            .try_with(Cell::get)
            .is_ok_and(|watched| watched == block.addr())
        {
            let _ = WATCHED_FREED.try_with(|freed| freed.set(true));
        }
    }
}

/// Pays for an operation: dropped, it records whether the watched block was
/// freed by then.
struct Funding;

impl Drop for Funding {
    fn drop(&mut self) {
        DROPPED.set(Some(WATCHED_FREED.get()));
    }
}

/// Watch the block `key`'s operation is kept in, which its debug output shows.
fn watch<T>(key: &Key<T>) {
    let shown = format!("{key:?}");
    let block = shown
        .strip_prefix("Key(")
        .and_then(|shown| shown.strip_suffix(')'))
        .and_then(|block| block.parse().ok())
        .expect("a key shows the address of its block");
    WATCHED.set(block);
    WATCHED_FREED.set(false);
    DROPPED.set(None);
}

fn push_and_wait<O: OpCode + 'static>(driver: &mut Proactor, op: O) -> BufResult<usize, O> {
    match driver.push(op) {
        PushEntry::Ready(res) => res,
        PushEntry::Pending(key) => wait(driver, key),
    }
}

fn wait<O: OpCode + 'static>(driver: &mut Proactor, mut key: Key<O>) -> BufResult<usize, O> {
    loop {
        match driver.pop(key) {
            PushEntry::Pending(k) => key = k,
            PushEntry::Ready(res) => break res,
        }
        driver.poll(None).unwrap();
    }
}

fn pipe(driver: &mut Proactor) -> (OwnedFd, OwnedFd) {
    let (_, op) = push_and_wait(driver, Pipe::new()).unwrap();
    op.into_inner()
}

/// Push a funded read of `r`, which has nothing to read yet, and watch its
/// block.
fn push_funded_read(driver: &mut Proactor, r: OwnedFd) -> Key<Read<Vec<u8>, OwnedFd>> {
    let key = match driver.try_push_funded(Read::new(r, Vec::with_capacity(16)), Funding) {
        Ok(PushEntry::Pending(key)) => key,
        Ok(PushEntry::Ready(_)) => panic!("the read completed with nothing to read"),
        Err(_) => panic!("the push was refused"),
    };
    watch(&key);
    key
}

#[test]
fn a_funded_op_drops_its_funding_after_its_block_is_freed() {
    let mut driver = Proactor::new().unwrap();
    let (r, w) = pipe(&mut driver);

    let key = push_funded_read(&mut driver, r);
    push_and_wait(&mut driver, Write::new(w, b"funded")).unwrap();
    assert_eq!(DROPPED.get(), None, "the funding is held with the op");

    let BufResult(res, _) = wait(&mut driver, key);
    assert_eq!(res.unwrap(), 6);
    assert_eq!(
        DROPPED.get(),
        Some(true),
        "the funding is dropped only after the op's block is freed"
    );
}

#[test]
fn a_cancel_token_outliving_its_op_keeps_the_funding() {
    let mut driver = Proactor::new().unwrap();
    let (r, w) = pipe(&mut driver);

    let key = push_funded_read(&mut driver, r);
    let token = driver.register_cancel(&key);
    push_and_wait(&mut driver, Write::new(w, b"funded")).unwrap();
    let BufResult(res, _) = wait(&mut driver, key);
    assert_eq!(res.unwrap(), 6);
    assert_eq!(
        DROPPED.get(),
        None,
        "the token keeps the op's block, and the funding with it"
    );

    drop(token);
    assert_eq!(
        DROPPED.get(),
        Some(true),
        "the funding is dropped only after the op's block is freed"
    );
}

#[cfg(io_uring)]
#[test]
fn a_cancelled_op_in_flight_keeps_its_funding_until_it_completes() {
    use std::time::Duration;

    let mut driver = Proactor::builder()
        .driver_type(compio_driver::DriverType::IoUring)
        .build()
        .unwrap();
    let (r, _w) = pipe(&mut driver);

    let key = push_funded_read(&mut driver, r);
    assert!(driver.cancel(key).is_none(), "the read is in flight");
    assert_eq!(
        DROPPED.get(),
        None,
        "the kernel still has the op, so its block and funding are held"
    );

    for _ in 0..100 {
        if DROPPED.get().is_some() {
            break;
        }
        driver.poll(Some(Duration::from_millis(10))).unwrap();
    }
    assert_eq!(
        DROPPED.get(),
        Some(true),
        "the funding is dropped only after the completed op's block is freed"
    );
}

#[test]
fn a_refused_push_allocates_nothing_and_drops_its_funding() {
    let mut driver = Proactor::new().unwrap();
    let (r, _w) = pipe(&mut driver);
    let op = Read::new(r, Vec::with_capacity(16));

    WATCHED.set(0);
    DROPPED.set(None);
    let before = LIVE.get();
    REFUSE.set(true);
    let refused = driver.try_push_funded(op, Funding);
    REFUSE.set(false);
    assert_eq!(LIVE.get(), before, "a refused push allocates nothing");
    assert_eq!(
        DROPPED.get(),
        Some(false),
        "a refused push drops its funding"
    );
    let Err(op) = refused else {
        panic!("the push was granted");
    };
    assert_eq!(
        op.into_inner().capacity(),
        16,
        "the op is handed back whole"
    );
}
