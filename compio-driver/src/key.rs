#![allow(dead_code)]

use std::{
    fmt::{self, Debug},
    hash::Hash,
    io,
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
    task::Waker,
};

use compio_buf::{BufResult, IntoInner};
use compio_send_wrapper::SendWrapper;
use thin_cell::unsync::{Inner, Ref, ThinCell, Weak};

use crate::{Carry, DriverType, Extra, OpCode, PushEntry, control::Carrier};

/// An operation with other needed information.
///
/// You should not use `RawOp` directly. Instead, use [`Key`] to manage the
/// reference-counted pointer to it.
#[repr(C)]
pub(crate) struct RawOp<M: ?Sized> {
    // Platform-specific extra data.
    //
    // - On Windows, it holds the `OVERLAPPED` buffer and a pointer to the driver.
    // - On Linux with `io_uring`, it holds the flags returned by kernel.
    // - On other platforms, it stores tracker for multi-fd `OpCode`s.
    //
    // Extra MUST be the first field to guarantee the layout for casting on windows. An invariant
    // on IOCP driver is that `RawOp` pointer is the same as `OVERLAPPED` pointer.
    extra: Extra,
    // The cancelled flag indicates the op has been cancelled.
    cancelled: bool,
    result: PushEntry<Option<Waker>, io::Result<usize>>,
    // What pays for the block the op is kept in. It is taken out by the last
    // reference to the block, strong or weak, and dropped after the block is
    // freed; dropping the op itself leaves it in place.
    funding: ManuallyDrop<Funding>,
    pub(crate) carrier: M,
}

/// How many words of room an operation's funding may take in the block the
/// operation is kept in; see [`Proactor::try_push_funded`].
///
/// [`Proactor::try_push_funded`]: crate::Proactor::try_push_funded
pub const OP_FUNDING_WORDS: usize = 3;

/// A funding kept in an operation's block, type-erased: dropped, it drops the
/// funding it was made from, if any.
pub(crate) struct Funding {
    value: MaybeUninit<[usize; OP_FUNDING_WORDS]>,
    drop: Option<unsafe fn(*mut ())>,
}

impl Funding {
    /// No funding.
    pub(crate) const NONE: Self = Self {
        value: MaybeUninit::uninit(),
        drop: None,
    };

    /// Keep `funding`, which must fit in [`OP_FUNDING_WORDS`] words, aligned
    /// to a word at most.
    pub(crate) fn new<C: 'static>(funding: C) -> Self {
        const {
            assert!(
                mem::size_of::<C>() <= mem::size_of::<[usize; OP_FUNDING_WORDS]>()
                    && mem::align_of::<C>() <= mem::align_of::<usize>(),
                "an operation's funding must fit in `OP_FUNDING_WORDS` words"
            )
        };
        if !mem::needs_drop::<C>() {
            // Nothing is given back when it goes
            mem::forget(funding);
            return Self::NONE;
        }
        let mut value = MaybeUninit::<[usize; OP_FUNDING_WORDS]>::uninit();
        // SAFETY: `C` fits the room and its alignment, as asserted above
        unsafe { value.as_mut_ptr().cast::<C>().write(funding) };
        Self {
            value,
            drop: Some(drop_funding::<C>),
        }
    }
}

/// Drops the funding of type `C` kept at `value`.
///
/// # Safety
///
/// `value` must point to a `C`, which is not used after.
unsafe fn drop_funding<C>(value: *mut ()) {
    unsafe { value.cast::<C>().drop_in_place() }
}

impl Drop for Funding {
    fn drop(&mut self) {
        if let Some(drop) = self.drop {
            // SAFETY: `drop` was made for the funding kept in `value`, which is
            // dropped once, here
            unsafe { drop(self.value.as_mut_ptr().cast()) }
        }
    }
}

impl<C: ?Sized> RawOp<C> {
    pub fn extra(&self) -> &Extra {
        &self.extra
    }

    pub fn extra_mut(&mut self) -> &mut Extra {
        &mut self.extra
    }

    #[cfg(io_uring)]
    pub fn wake_by_ref(&mut self) {
        if let PushEntry::Pending(Some(w)) = &self.result {
            w.wake_by_ref();
        }
    }
}

#[cfg(io_uring)]
impl<C: crate::Carry + ?Sized> RawOp<C> {
    pub fn create_entry<const FALLBACK: bool>(&mut self) -> crate::OpEntry {
        if FALLBACK {
            self.carrier.create_entry_fallback().with_extra(&self.extra)
        } else {
            self.carrier.create_entry().with_extra(&self.extra)
        }
    }
}

#[cfg(windows)]
impl<C: crate::Carry + ?Sized> RawOp<C> {
    /// Call [`OpCode::operate`] and assume that it is not an overlapped op,
    /// which means it never returns [`Poll::Pending`].
    ///
    /// [`Poll::Pending`]: std::task::Poll::Pending
    pub fn operate_blocking(&mut self) -> io::Result<usize> {
        use std::{panic::AssertUnwindSafe, task::Poll};

        use crate::panic::catch_unwind_io;

        let optr = self.extra_mut().optr();
        catch_unwind_io(AssertUnwindSafe(|| unsafe {
            match self.carrier.operate(optr.cast()) {
                Poll::Pending => unreachable!("this operation is not overlapped"),
                Poll::Ready(res) => res,
            }
        }))
    }
}

impl<C: ?Sized> Debug for RawOp<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawOp")
            .field("extra", &self.extra)
            .field("cancelled", &self.cancelled)
            .field("result", &self.result)
            .field("Carrier", &"<...>")
            .finish()
    }
}

/// A typed wrapper for key of Ops submitted into driver.
#[repr(transparent)]
pub struct Key<T> {
    erased: ErasedKey,
    _p: std::marker::PhantomData<T>,
}

/// A type-erased reference-counted pointer to an operation.
///
/// Internally, it uses [`ThinCell`] to manage the reference count and borrowing
/// state. It provides methods to manipulate the underlying operation, such as
/// setting results, checking completion status, and cancelling the operation.
///
/// The last reference to the operation's block, this or a [`WeakKey`], drops
/// the funding of the operation after the block is freed.
#[derive(Clone)]
#[repr(transparent)]
pub struct ErasedKey {
    inner: ManuallyDrop<ThinCell<RawOp<dyn Carry>>>,
}

/// A weak reference of [`ErasedKey`].
#[derive(Clone)]
pub(crate) struct WeakKey {
    inner: ManuallyDrop<Weak<RawOp<dyn Carry>>>,
    // Where the funding of the operation is kept in its block, for the last
    // reference to take.
    funding: NonNull<ManuallyDrop<Funding>>,
}

impl<T> Clone for Key<T> {
    fn clone(&self) -> Self {
        Self {
            erased: self.erased.clone(),
            _p: std::marker::PhantomData,
        }
    }
}

impl<T> Debug for Key<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Key({})", self.erased.inner.as_ptr() as usize)
    }
}

impl<T> Key<T> {
    pub(crate) fn into_raw(self) -> usize {
        self.erased.into_raw()
    }

    pub(crate) fn erase(self) -> ErasedKey {
        self.erased
    }
}

impl<T: OpCode> Key<T> {
    /// Take the inner result if it is completed.
    ///
    /// # Panics
    ///
    /// Panics if the result is not ready or the `Key` is not unique (multiple
    /// references or borrowed).
    pub(crate) fn take_result(self) -> BufResult<usize, T> {
        // SAFETY: `Key` invariant guarantees that `T` is the actual concrete
        // type.
        unsafe { self.erased.take_result::<T>() }
    }
}

impl<T: OpCode + 'static> Key<T> {
    /// Create [`RawOp`] and get the [`Key`] to it.
    pub(crate) fn new(op: T, extra: impl Into<Extra>, driver_ty: DriverType) -> Self {
        let erased = ErasedKey::new(op, extra.into(), driver_ty);

        Self {
            erased,
            _p: std::marker::PhantomData,
        }
    }

    /// Create [`RawOp`] funded by `funding` and get the [`Key`] to it, unless
    /// the allocator refuses the block it is kept in: refused, nothing is
    /// allocated, `funding` is dropped and `op` is handed back.
    pub(crate) fn try_new_funded(
        op: T,
        extra: impl Into<Extra>,
        driver_ty: DriverType,
        funding: Funding,
    ) -> Result<Self, T> {
        let erased = ErasedKey::try_new_funded(op, extra.into(), driver_ty, funding)?;

        Ok(Self {
            erased,
            _p: std::marker::PhantomData,
        })
    }

    pub(crate) fn set_extra(&self, extra: impl Into<Extra>) {
        self.borrow().extra = extra.into();
    }
}

impl<T> Deref for Key<T> {
    type Target = ErasedKey;

    fn deref(&self) -> &Self::Target {
        &self.erased
    }
}

impl<T> DerefMut for Key<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.erased
    }
}

impl PartialEq for ErasedKey {
    fn eq(&self, other: &Self) -> bool {
        self.inner.ptr_eq(&other.inner)
    }
}

impl Eq for ErasedKey {}

impl Hash for ErasedKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.inner.as_ptr() as usize).hash(state)
    }
}

impl Unpin for ErasedKey {}

impl ErasedKey {
    /// Create [`RawOp`] and get the [`ErasedKey`] to it.
    pub(crate) fn new<T: OpCode + 'static>(op: T, extra: Extra, driver_ty: DriverType) -> Self {
        let raw_op = RawOp {
            extra,
            cancelled: false,
            result: PushEntry::Pending(None),
            funding: ManuallyDrop::new(Funding::NONE),
            carrier: Carrier::new(op, driver_ty),
        };
        Self::from_cell(ThinCell::new(raw_op))
    }

    /// Create [`RawOp`] funded by `funding` and get the [`ErasedKey`] to it,
    /// unless the allocator refuses the block it is kept in: refused, nothing
    /// is allocated, `funding` is dropped and `op` is handed back.
    pub(crate) fn try_new_funded<T: OpCode + 'static>(
        op: T,
        extra: Extra,
        driver_ty: DriverType,
        funding: Funding,
    ) -> Result<Self, T> {
        let raw_op = RawOp {
            extra,
            cancelled: false,
            result: PushEntry::Pending(None),
            funding: ManuallyDrop::new(funding),
            carrier: Carrier::new(op, driver_ty),
        };
        match ThinCell::try_new(raw_op) {
            Ok(inner) => Ok(Self::from_cell(inner)),
            Err(raw_op) => {
                let RawOp {
                    funding, carrier, ..
                } = raw_op;
                drop(ManuallyDrop::into_inner(funding));
                Err(carrier.into_inner())
            }
        }
    }

    /// Initialize the carrier of a just created cell, now pinned in its block.
    fn from_cell<T: OpCode + 'static>(mut inner: ThinCell<RawOp<Carrier<T>>>) -> Self {
        // SAFETY:
        // - ThinCell is just created, there will be no shared owner or borrower
        // - Carrier is being pinned by ThinCell, it will have a stable address
        //   until move out
        unsafe { inner.borrow_unchecked().carrier.init() };
        Self {
            inner: ManuallyDrop::new(unsafe {
                inner.unsize(|p| p as *const Inner<RawOp<dyn Carry>>)
            }),
        }
    }

    /// Create from `user_data` pointer.
    ///
    /// # Safety
    ///
    /// `user_data` must be a valid pointer to `RawOp<dyn OpCode>` previously
    /// created by [`Key::into_raw`].
    pub(crate) unsafe fn from_raw(user_data: usize) -> Self {
        let inner = ManuallyDrop::new(unsafe { ThinCell::from_raw(user_data as *mut ()) });
        Self { inner }
    }

    /// Create from `Overlapped` pointer.
    ///
    /// # Safety
    ///
    /// `optr` must be a valid pointer to `Overlapped` stored in `Extra` of
    /// `RawOp<dyn OpCode>`.
    #[cfg(windows)]
    pub(crate) unsafe fn from_optr(optr: *mut crate::sys::Overlapped) -> Self {
        let ptr = unsafe { optr.cast::<usize>().offset(-2).cast() };
        let inner = ManuallyDrop::new(unsafe { ThinCell::from_raw(ptr) });
        Self { inner }
    }

    /// Leak self into a pointer to `Overlapped`.
    #[cfg(windows)]
    pub(crate) fn into_optr(self) -> *mut crate::sys::Overlapped {
        let this = ManuallyDrop::new(self);
        unsafe { this.inner.as_ptr().cast_mut().cast::<usize>().add(2).cast() }
    }

    /// Get a weak reference to the allocation.
    ///
    /// It will not prevent dropping [`RawOp`] or cause panic when calling
    /// [`take_result`], and can be upgrade back when needed.
    ///
    /// [`take_result`]: Self::take_result
    pub(crate) fn downgrade(&self) -> WeakKey {
        // The funding's place is derived from the pointer to the whole block,
        // as it is used once the op is gone.
        let block = self.inner.as_ptr();
        let at = (&raw const self.borrow().funding).addr() - block.addr();
        let funding = block.cast_mut().cast::<u8>().wrapping_add(at).cast();
        WeakKey {
            inner: ManuallyDrop::new(self.inner.downgrade()),
            // SAFETY: a place in a live block is not null
            funding: unsafe { NonNull::new_unchecked(funding) },
        }
    }

    /// Get the pointer as `user_data`.
    ///
    /// **Do not** call [`from_raw`](Self::from_raw) on the returned value of
    /// this method.
    pub(crate) fn as_raw(&self) -> usize {
        self.inner.as_ptr() as _
    }

    /// Leak self and get the pointer as `user_data`.
    pub(crate) fn into_raw(self) -> usize {
        let this = ManuallyDrop::new(self);
        this.inner.as_ptr() as _
    }

    #[inline]
    pub(crate) fn borrow(&self) -> Ref<'_, RawOp<dyn Carry>> {
        self.inner.borrow()
    }

    /// Set the `cancelled` flag, returning whether it was already cancelled.
    pub(crate) fn set_cancelled(&self) -> bool {
        let mut op = self.borrow();
        mem::replace(&mut op.cancelled, true)
    }

    /// Whether the op is completed.
    pub(crate) fn has_result(&self) -> bool {
        self.borrow().result.is_ready()
    }

    /// Whether the key is uniquely owned.
    pub(crate) fn is_unique(&self) -> bool {
        self.inner.count() == 1
    }

    /// Complete the op and wake up the future if a waker is set.
    pub(crate) fn set_result(&self, res: io::Result<usize>) {
        let mut this = self.borrow();
        {
            let RawOp { extra, carrier, .. } = &mut *this;
            unsafe { crate::sys::Carry::set_result(carrier, &res, extra) };
        }
        if let PushEntry::Pending(Some(w)) =
            std::mem::replace(&mut this.result, PushEntry::Ready(res))
        {
            w.wake();
        }
    }

    /// Swap the inner [`Extra`] with the provided one, returning the previous
    /// value.
    pub(crate) fn swap_extra(&self, extra: Extra) -> Extra {
        std::mem::replace(&mut self.borrow().extra, extra)
    }

    /// Set waker of the current future.
    pub(crate) fn set_waker(&self, waker: &Waker) {
        let PushEntry::Pending(w) = &mut self.borrow().result else {
            return;
        };

        if w.as_ref().is_some_and(|w| w.will_wake(waker)) {
            return;
        }

        *w = Some(waker.clone());
    }

    /// Take the inner result if it is completed.
    ///
    /// # Safety
    ///
    /// `T` must be the actual concrete type of the `Key`.
    ///
    /// # Panics
    ///
    /// Panics if the result is not ready or the `Key` is not unique (multiple
    /// references or borrowed).
    unsafe fn take_result<T: OpCode>(self) -> BufResult<usize, T> {
        let mut this = ManuallyDrop::new(self);
        // Taken first, dropped after the block is freed, below
        let funding = this.take_last_funding();
        // SAFETY: The cell is taken once, here, and `this` is not dropped.
        let inner = unsafe { ManuallyDrop::take(&mut this.inner) };
        // SAFETY: Caller guarantees that `T` is the actual concrete type.
        let inner = unsafe { inner.downcast_unchecked::<RawOp<Carrier<T>>>() };
        let op = inner.try_unwrap().map_err(|_| ()).expect("Key not unique");
        let res = op.result.take_ready().expect("Result not ready");
        let res = BufResult(res, op.carrier.into_inner());
        drop(funding);
        res
    }

    /// Take the funding out of the op's block if this is the last reference to
    /// the block, strong or weak: the block is freed as it goes, and the
    /// funding, paying for it, is dropped only after. Otherwise the funding is
    /// left in the block, for whichever reference is last, and none is taken.
    fn take_last_funding(&mut self) -> Funding {
        if self.inner.strong_count() != 1 || self.inner.weak_count() != 1 {
            return Funding::NONE;
        }
        // SAFETY: This is the last reference, so nothing borrows the op, and
        // the funding is taken once, by the last reference.
        unsafe { ManuallyDrop::take(&mut self.inner.borrow_unchecked().funding) }
    }

    /// Unsafely freeze the `Key` by bypassing borrow flag of [`ThinCell`],
    /// preventing it from being dropped and unconditionally expose the
    /// underlying `RawOp<dyn OpCode>`.
    ///
    /// # Safety
    /// - During the time the [`FrozenKey`] is alive, no other references to the
    ///   underlying `RawOp<dyn OpCode>` is used.
    /// - One must not touch [`ThinCell`]'s internal state at all, as `Cell` is
    ///   strictly single-threaded. This means no borrowing, no cloning, no
    ///   dropping, etc.
    pub(crate) unsafe fn freeze(self) -> FrozenKey {
        FrozenKey {
            inner: ManuallyDrop::new(self),
            thread_id: SendWrapper::new(()),
        }
    }
}

impl Drop for ErasedKey {
    fn drop(&mut self) {
        let funding = self.take_last_funding();
        // SAFETY: The cell is dropped once, here, and not used after.
        unsafe { ManuallyDrop::drop(&mut self.inner) };
        drop(funding);
    }
}

impl Debug for ErasedKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ErasedKey({})", self.inner.as_ptr() as usize)
    }
}

impl WeakKey {
    pub(crate) fn upgrade(&self) -> Option<ErasedKey> {
        Some(ErasedKey {
            inner: ManuallyDrop::new(self.inner.upgrade()?),
        })
    }

    pub(crate) fn as_ptr(&self) -> *const () {
        self.inner.as_ptr()
    }
}

impl Drop for WeakKey {
    fn drop(&mut self) {
        // The last reference: the op was dropped, leaving its funding in the
        // block, which goes with this reference.
        let funding = if self.inner.strong_count() == 0 && self.inner.weak_count() == 1 {
            // SAFETY: The block is still allocated, as this reference holds it,
            // and nothing else refers to it; the funding is taken once, by the
            // last reference.
            unsafe { ManuallyDrop::take(self.funding.as_mut()) }
        } else {
            Funding::NONE
        };
        // SAFETY: The weak reference is dropped once, here, and not used after.
        unsafe { ManuallyDrop::drop(&mut self.inner) };
        drop(funding);
    }
}

impl PartialEq for WeakKey {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.inner.as_ptr(), other.inner.as_ptr())
    }
}

impl Eq for WeakKey {}

impl Hash for WeakKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.inner.as_ptr() as usize).hash(state)
    }
}

impl Debug for WeakKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(upgraded) = self.inner.upgrade() {
            Debug::fmt(&upgraded, f)
        } else {
            write!(f, "(Dropped)")
        }
    }
}

/// A frozen view into a [`Key`].
///
/// It's guaranteed to have [`ErasedKey`] as the first field.
#[repr(C)]
pub(crate) struct FrozenKey {
    inner: ManuallyDrop<ErasedKey>,
    thread_id: SendWrapper<()>,
}

impl FrozenKey {
    pub fn as_mut(&mut self) -> &mut RawOp<dyn Carry> {
        unsafe { self.inner.inner.borrow_unchecked() }
    }

    pub fn into_inner(self) -> ErasedKey {
        let mut this = ManuallyDrop::new(self);
        unsafe { ManuallyDrop::take(&mut this.inner) }
    }
}

impl Drop for FrozenKey {
    fn drop(&mut self) {
        if self.thread_id.valid() {
            unsafe { ManuallyDrop::drop(&mut self.inner) }
        }
    }
}

unsafe impl Send for FrozenKey {}
unsafe impl Sync for FrozenKey {}

/// A temporary view into a [`Key`].
///
/// It is mainly used in the driver to avoid accidentally decreasing the
/// reference count of the `Key` when the driver is not completed and may still
/// emit event with the `user_data`.
pub(crate) struct BorrowedKey(ManuallyDrop<ErasedKey>);

impl BorrowedKey {
    pub unsafe fn from_raw(user_data: usize) -> Self {
        let key = unsafe { ErasedKey::from_raw(user_data) };
        Self(ManuallyDrop::new(key))
    }

    pub fn upgrade(self) -> ErasedKey {
        ManuallyDrop::into_inner(self.0)
    }
}

impl Deref for BorrowedKey {
    type Target = ErasedKey;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
