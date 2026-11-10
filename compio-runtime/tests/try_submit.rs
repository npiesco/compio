//! Assert that a funded submission dropped while its operation is in flight
//! keeps its funding until the block the operation is kept in is freed, as
//! the runtime completes it, measured with a counting allocator.
#![cfg(unix)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    future::{Future, poll_fn},
    os::fd::OwnedFd,
    rc::Rc,
    task::Poll,
};

use compio_buf::{BufResult, IntoInner};
use compio_driver::op::{Pipe, Read, Write};

/// Notes the largest block allocated while recording, and when the watched
/// block is freed.
struct Counting;

#[global_allocator]
static ALLOCATOR: Counting = Counting;

std::thread_local! {
    static RECORDING: Cell<bool> = const { Cell::new(false) };
    /// The size and address of the largest block allocated while recording.
    static LARGEST: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    static WATCHED: Cell<usize> = const { Cell::new(0) };
    static WATCHED_FREED: Cell<bool> = const { Cell::new(false) };
    /// Whether the watched block was freed when a [`Funding`] was last
    /// dropped.
    static DROPPED: Cell<Option<bool>> = const { Cell::new(None) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let block = unsafe { System.alloc(layout) };
        if !block.is_null() && RECORDING.try_with(Cell::get).unwrap_or(false) {
            let _ = LARGEST.try_with(|largest| {
                if layout.size() > largest.get().0 {
                    largest.set((layout.size(), block.addr()));
                }
            });
        }
        block
    }

    unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
        unsafe { System.dealloc(block, layout) };
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

async fn pipe() -> (OwnedFd, OwnedFd) {
    let BufResult(res, op) = compio_runtime::submit(Pipe::new()).await;
    res.unwrap();
    op.into_inner()
}

#[test]
fn a_dropped_try_submit_keeps_its_funding_until_its_op_block_is_freed() {
    compio_runtime::Runtime::new().unwrap().block_on(async {
        let (r, _w) = pipe().await;
        // Written to drive the runtime, apart from the pipe the read waits on.
        let (_r, w) = pipe().await;
        let w = Rc::new(w);
        // Warm the driver up, so the read's block is the largest it allocates.
        let warm = compio_runtime::try_submit_funded(Write::new(w.clone(), b"w"), ());
        let Ok(BufResult(res, _)) = warm.await else {
            panic!("the warming write was refused");
        };
        res.unwrap();

        let mut read = Box::pin(compio_runtime::try_submit_funded(
            Read::new(r, Vec::with_capacity(16)),
            Funding,
        ));
        DROPPED.set(None);
        LARGEST.set((0, 0));
        RECORDING.set(true);
        let pending = poll_fn(|cx| Poll::Ready(read.as_mut().poll(cx).is_pending())).await;
        RECORDING.set(false);
        assert!(pending, "the read is pushed, with nothing to read");
        WATCHED.set(LARGEST.get().1);
        WATCHED_FREED.set(false);

        drop(read);
        assert_ne!(
            DROPPED.get(),
            Some(false),
            "dropped in flight, the funding is held with the op's block"
        );

        for _ in 0..100 {
            if DROPPED.get().is_some() {
                break;
            }
            let BufResult(res, _) = compio_runtime::submit(Write::new(w.clone(), b"x")).await;
            res.unwrap();
        }
        assert_eq!(
            DROPPED.get(),
            Some(true),
            "the funding is dropped only after the completed op's block is freed"
        );
    });
}
