#![allow(missing_docs)]

use core::fmt;
use core::iter::{FromIterator, FusedIterator};
use core::mem::{self, swap, ManuallyDrop};
use core::ops::{Deref, DerefMut};
use core::ptr;

use std::borrow::Borrow;
use std::collections::TryReserveError;
use std::slice;
use std::vec::{self, Vec};

#[derive(Debug, Clone)]
pub struct HeapNode<K, V> {
    pub key: K,
    pub value: V,
}

impl<K: PartialEq, V> PartialEq for HeapNode<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.key.eq(&other.key)
    }
}

impl<K: Eq, V> Eq for HeapNode<K, V> {}

impl<K: PartialOrd + Eq, V> PartialOrd for HeapNode<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.key.partial_cmp(&other.key)
    }
}

impl<K: Ord, V> Ord for HeapNode<K, V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl<K: Copy, V: Copy> Copy for HeapNode<K, V> {}

/// A priority queue implemented with a binary heap.
///
/// This will be a max-heap.
///
/// It is a logic error for an item to be modified in such a way that the
/// item's ordering relative to any other item, as determined by the [`Ord`]
/// trait, changes while it is in the heap. This is normally only possible
/// through [`Cell`], [`RefCell`], global state, I/O, or unsafe code. The
/// behavior resulting from such a logic error is not specified (it
/// could include panics, incorrect results, aborts, memory leaks, or
/// non-termination) but will not be undefined behavior.
///
/// # Examples
///
/// ```
/// use kv_heap::{HeapNode, KeyValueHeap};
///
/// // Type inference lets us omit an explicit type signature (which
/// // would be `KeyValueHeap<i32>` in this example).
/// let mut heap = KeyValueHeap::new();
///
/// // We can use peek to look at the next item in the heap. In this case,
/// // there's no items in there yet so we get None.
/// assert_eq!(heap.peek(), None);
///
/// // Let's add some scores...
/// heap.push(1, "Bob");
/// heap.push(5, "Alice");
/// heap.push(2, "Eve");
///
/// // Now peek shows the most important item in the heap.
/// assert_eq!(heap.peek(), Some(&HeapNode { key: 5, value: "Alice" }));
///
/// // We can check the length of a heap.
/// assert_eq!(heap.len(), 3);
///
/// // We can iterate over the items in the heap, although they are returned in
/// // a random order.
/// for x in &heap {
///     println!("{x}");
/// }
///
/// // If we instead pop these scores, they should come back in order.
/// assert_eq!(heap.pop().map(|e| e.value), Some("Alice"));
/// assert_eq!(heap.pop().map(|e| e.value), Some("Eve"));
/// assert_eq!(heap.pop().map(|e| e.value), Some("Bob"));
/// assert_eq!(heap.pop().map(|e| e.value), None);
///
/// // We can clear the heap of any remaining items.
/// heap.clear();
///
/// // The heap should now be empty.
/// assert!(heap.is_empty())
/// ```
///
/// A `KeyValueHeap` with a known list of items can be initialized from an array:
///
/// ```
/// use kv_heap::KeyValueHeap;
///
/// let heap = KeyValueHeap::<i32, &str>::from([(1, "Bob"), (5, "Alice"), (2, "Eve")]);
/// ```
///
/// ## Min-heap
///
/// Either [`core::cmp::Reverse`] or a custom [`Ord`] implementation can be used to
/// make `KeyValueHeap` a min-heap. This makes `heap.pop()` return the smallest
/// value instead of the greatest one.
///
/// ```
/// use kv_heap::KeyValueHeap;
/// use std::cmp::Reverse;
///
/// let mut heap = KeyValueHeap::new();
///
/// // Wrap values in `Reverse`
/// heap.push(Reverse(1), "Bob");
/// heap.push(Reverse(5), "Alice");
/// heap.push(Reverse(2), "Eve");
///
/// // If we pop these scores now, they should come back in the reverse order.
/// assert_eq!(heap.pop().map(|e| e.key), Some(Reverse(1))); // Bob
/// assert_eq!(heap.pop().map(|e| e.key), Some(Reverse(2))); // Eve
/// assert_eq!(heap.pop().map(|e| e.key), Some(Reverse(5))); // Alice
/// assert_eq!(heap.pop().map(|e| e.key), None);
/// ```
///
/// # Time complexity
///
/// | [push]  | [pop]         | [peek]/[peek\_mut] |
/// |---------|---------------|--------------------|
/// | *O*(1)~ | *O*(log(*n*)) | *O*(1)             |
///
/// The value for `push` is an expected cost; the method documentation gives a
/// more detailed analysis.
///
/// [`core::cmp::Reverse`]: core::cmp::Reverse
/// [`Ord`]: core::cmp::Ord
/// [`Cell`]: core::cell::Cell
/// [`RefCell`]: core::cell::RefCell
/// [push]: KeyValueHeap::push
/// [pop]: KeyValueHeap::pop
/// [peek]: KeyValueHeap::peek
/// [peek\_mut]: KeyValueHeap::peek_mut

pub struct KeyValueHeap<K, V> {
    data: Vec<HeapNode<K, V>>,
}

/// Structure wrapping a mutable reference to the greatest item on a
/// `KeyValueHeap`.
///
/// This `struct` is created by the [`peek_mut`] method on [`KeyValueHeap`]. See
/// its documentation for more.
///
/// [`peek_mut`]: KeyValueHeap::peek_mut

pub struct PeekMut<'a, K: 'a + Ord, V: 'a> {
    heap: &'a mut KeyValueHeap<K, V>,
    sift: bool,
}

impl<K: Ord + fmt::Debug, V: fmt::Debug> fmt::Debug for PeekMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PeekMut").field(&self.heap.data[0]).finish()
    }
}

impl<K: Ord, V> Drop for PeekMut<'_, K, V> {
    fn drop(&mut self) {
        if self.sift {
            // SAFETY: PeekMut is only instantiated for non-empty heaps.
            unsafe { self.heap.sift_down(0) };
        }
    }
}

impl<K: Ord, V> Deref for PeekMut<'_, K, V> {
    type Target = HeapNode<K, V>;
    fn deref(&self) -> &Self::Target {
        debug_assert!(!self.heap.is_empty());
        // SAFE: PeekMut is only instantiated for non-empty heaps
        unsafe { self.heap.data.get_unchecked(0) }
    }
}

impl<K: Ord, V> DerefMut for PeekMut<'_, K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        debug_assert!(!self.heap.is_empty());
        self.sift = true;
        // SAFE: PeekMut is only instantiated for non-empty heaps
        unsafe { self.heap.data.get_unchecked_mut(0) }
    }
}

impl<'a, K: Ord, V> PeekMut<'a, K, V> {
    /// Removes the peeked value from the heap and returns it.

    pub fn pop(mut this: PeekMut<'a, K, V>) -> HeapNode<K, V> {
        let value = this.heap.pop().unwrap();
        this.sift = false;
        value
    }
}

impl<K: Clone, V: Clone> Clone for KeyValueHeap<K, V> {
    fn clone(&self) -> Self {
        KeyValueHeap {
            data: self.data.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.data.clone_from(&source.data);
    }
}

impl<K: Ord, V> Default for KeyValueHeap<K, V> {
    /// Creates an empty `KeyValueHeap<K, V>`.
    #[inline]
    fn default() -> KeyValueHeap<K, V> {
        KeyValueHeap::new()
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for KeyValueHeap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.data.iter()).finish()
    }
}

#[allow(unused_unsafe)]
impl<K: Ord, V> KeyValueHeap<K, V> {
    /// Creates an empty `KeyValueHeap` as a max-heap.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let mut heap = KeyValueHeap::new();
    /// heap.push(4, "Steven");
    /// ```
    #[must_use]
    pub fn new() -> KeyValueHeap<K, V> {
        KeyValueHeap { data: vec![] }
    }

    /// Creates a new heap using an iteratable object of keys and a mapping to associated values
    /// as source.
    ///
    /// # Examokes
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::{KeyValueHeap, HeapNode};
    ///
    /// let mut heap = KeyValueHeap::new_with_mapping(
    ///     [1, 2, 3, 4, 5, 6, 7],
    ///     |prio| format!("Prio #{} Object", prio)
    /// );
    ///
    /// assert_eq!(heap.pop(), Some(HeapNode { key: 7, value: "Prio #7 Object".to_string() }));
    ///
    /// heap.change_key("Prio #3 Object".to_string(), 10_000);
    /// assert_eq!(heap.pop(), Some(HeapNode { key: 10_000, value: "Prio #3 Object".to_string() }));
    /// ```
    #[must_use]
    pub fn new_with_mapping<F>(iter: impl IntoIterator<Item = K>, mapping: F) -> KeyValueHeap<K, V>
    where
        F: Fn(&K) -> V,
    {
        let iter = iter.into_iter();
        let mut data = Vec::with_capacity(iter.size_hint().0);
        for key in iter {
            data.push(HeapNode {
                value: mapping(&key),
                key,
            })
        }

        let mut heap = Self { data };
        heap.rebuild();
        heap
    }

    /// Creates an empty `KeyValueHeap` with a specific capacity.
    /// This preallocates enough memory for `capacity` elements,
    /// so that the `KeyValueHeap` does not have to be reallocated
    /// until it contains at least that many values.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let mut heap = KeyValueHeap::with_capacity(10);
    /// heap.push(4, "Eve");
    /// ```
    #[must_use]
    pub fn with_capacity(capacity: usize) -> KeyValueHeap<K, V> {
        KeyValueHeap {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Returns a mutable reference to the greatest item in the binary heap, or
    /// `None` if it is empty.
    ///
    /// Note: If the `PeekMut` value is leaked, the heap may be in an
    /// inconsistent state.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let mut heap = KeyValueHeap::new();
    /// assert!(heap.peek_mut().is_none());
    ///
    /// heap.push(1, "Bob");
    /// heap.push(5, "Alice");
    /// heap.push(2, "Eve");
    /// {
    ///     let mut val = heap.peek_mut().unwrap();
    ///     val.value = "Steven";
    /// }
    /// assert_eq!(heap.peek().map(|e| e.value), Some("Steven"));
    /// ```
    ///
    /// # Time complexity
    ///
    /// If the item is modified then the worst case time complexity is *O*(log(*n*)),
    /// otherwise it's *O*(1).
    pub fn peek_mut(&mut self) -> Option<PeekMut<'_, K, V>> {
        if self.is_empty() {
            None
        } else {
            Some(PeekMut {
                heap: self,
                sift: false,
            })
        }
    }

    /// Removes the greatest item from the binary heap and returns it, or `None` if it
    /// is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let mut heap = KeyValueHeap::from([1, 3]);
    ///
    /// assert_eq!(heap.pop().map(|e| e.value), Some(3));
    /// assert_eq!(heap.pop().map(|e| e.value), Some(1));
    /// assert_eq!(heap.pop().map(|e| e.value), None);
    /// ```
    ///
    /// # Time complexity
    ///
    /// The worst case cost of `pop` on a heap containing *n* elements is *O*(log(*n*)).
    pub fn pop(&mut self) -> Option<HeapNode<K, V>> {
        self.data.pop().map(|mut item| {
            if !self.is_empty() {
                swap(&mut item, &mut self.data[0]);
                // SAFETY: !self.is_empty() means that self.len() > 0
                unsafe { self.sift_down_to_bottom(0) };
            }

            item
        })
    }

    pub fn check_integrity(&self) -> bool {
        let mut indices = vec![0];
        while let Some(i) = indices.pop() {
            if i >= self.data.len() {
                continue;
            }
            let left = 2 * i + 1;
            let right = 2 * i + 2;

            let parent_element = &self.data[i];
            if left < self.data.len() && self.data[left].key >= parent_element.key {
                return false;
            }
            if right < self.data.len() && self.data[right].key >= parent_element.key {
                return false;
            }
        }
        true
    }

    /// Pushes an item onto the binary heap.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::{KeyValueHeap, HeapNode};
    /// let mut heap = KeyValueHeap::new();
    /// heap.push(3, "Bob");
    /// heap.push(5, "Alice");
    /// heap.push(1, "Eve");
    ///
    /// assert_eq!(heap.len(), 3);
    /// assert_eq!(heap.peek(), Some(&HeapNode { key: 5, value: "Alice" }));
    /// ```
    ///
    /// # Time complexity
    ///
    /// The expected cost of `push`, averaged over every possible ordering of
    /// the elements being pushed, and over a sufficiently large number of
    /// pushes, is *O*(1). This is the most meaningful cost metric when pushing
    /// elements that are *not* already in any sorted pattern.
    ///
    /// The time complexity degrades if elements are pushed in predominantly
    /// ascending order. In the worst case, elements are pushed in ascending
    /// sorted order and the amortized cost per push is *O*(log(*n*)) against a heap
    /// containing *n* elements.
    ///
    /// The worst case cost of a *single* call to `push` is *O*(*n*). The worst case
    /// occurs when capacity is exhausted and needs a resize. The resize cost
    /// has been amortized in the previous figures.
    pub fn push(&mut self, key: K, value: V) {
        let old_len = self.len();
        self.data.push(HeapNode { key, value });
        // SAFETY: Since we pushed a new item it means that
        //  old_len = self.len() - 1 < self.len()
        unsafe { self.sift_up(0, old_len) };
    }

    ///
    /// This function changes the key of the first element with the given key.
    /// Caller must gurantee that key exists.
    ///
    pub fn change_key<U>(&mut self, value: U, new_key: K)
    where
        U: Borrow<V>,
        V: PartialEq,
    {
        let (index, _) = self
            .data
            .iter()
            .enumerate()
            .find(|(_, node)| node.value == *value.borrow())
            .unwrap();

        unsafe { self.change_key_by_index(index, new_key) }
    }

    unsafe fn change_key_by_index(&mut self, index: usize, new_key: K) {
        if new_key <= self.data[index].key {
            self.data[index].key = new_key;

            // Move to top
            let mut hole = Hole::new(&mut self.data, index);
            while hole.pos() != 0 {
                let parent = (hole.pos() - 1) / 2;
                // SAFETY: Same as above
                unsafe { hole.move_to(parent) };
            }

            drop(hole);
            let node = self.pop().unwrap();
            let old_len = self.data.len();

            self.data.push(node);
            self.sift_up(0, old_len);
        } else {
            // Change key towards root.
            self.data[index].key = new_key;
            self.sift_up(0, index);
        }
    }

    /// Consumes the `KeyValueHeap` and returns a vector in sorted
    /// (ascending) order.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    ///
    /// let mut heap = KeyValueHeap::from([1, 2, 4, 5, 7]);
    /// heap.push(6, 6);
    /// heap.push(3, 3);
    ///
    /// let vec = heap.into_sorted_vec();
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6, 7]);
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    pub fn into_sorted_vec(mut self) -> Vec<V> {
        let mut end = self.len();
        while end > 1 {
            end -= 1;
            // SAFETY: `end` goes from `self.len() - 1` to 1 (both included),
            //  so it's always a valid index to access.
            //  It is safe to access index 0 (i.e. `ptr`), because
            //  1 <= end < self.len(), which means self.len() >= 2.
            unsafe {
                let ptr = self.data.as_mut_ptr();
                ptr::swap(ptr, ptr.add(end));
            }
            // SAFETY: `end` goes from `self.len() - 1` to 1 (both included) so:
            //  0 < 1 <= end <= self.len() - 1 < self.len()
            //  Which means 0 < end and end < self.len().
            unsafe { self.sift_down_range(0, end) };
        }
        self.into_vec()
    }

    // The implementations of sift_up and sift_down use unsafe blocks in
    // order to move an element out of the vector (leaving behind a
    // hole), shift along the others and move the removed element back into the
    // vector at the final location of the hole.
    // The `Hole` type is used to represent this, and make sure
    // the hole is filled back at the end of its scope, even on panic.
    // Using a hole reduces the constant factor compared to using swaps,
    // which involves twice as many moves.

    /// # Safety
    ///
    /// The caller must guarantee that `pos < self.len()`.
    unsafe fn sift_up(&mut self, start: usize, pos: usize) -> usize {
        // Take out the value at `pos` and create a hole.
        // SAFETY: The caller guarantees that pos < self.len()
        let mut hole = unsafe { Hole::new(&mut self.data, pos) };

        while hole.pos() > start {
            let parent = (hole.pos() - 1) / 2;

            // SAFETY: hole.pos() > start >= 0, which means hole.pos() > 0
            //  and so hole.pos() - 1 can't underflow.
            //  This guarantees that parent < hole.pos() so
            //  it's a valid index and also != hole.pos().
            if &hole.element().key <= unsafe { &hole.get(parent).key } {
                break;
            }

            // SAFETY: Same as above
            unsafe { hole.move_to(parent) };
        }

        hole.pos()
    }

    /// Take an element at `pos` and move it down the heap,
    /// while its children are larger.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `pos < end <= self.len()`.
    unsafe fn sift_down_range(&mut self, pos: usize, end: usize) {
        // SAFETY: The caller guarantees that pos < end <= self.len().
        let mut hole = unsafe { Hole::new(&mut self.data, pos) };
        let mut child = 2 * hole.pos() + 1;

        // Loop invariant: child == 2 * hole.pos() + 1.
        while child <= end.saturating_sub(2) {
            // compare with the greater of the two children
            // SAFETY: child < end - 1 < self.len() and
            //  child + 1 < end <= self.len(), so they're valid indexes.
            //  child == 2 * hole.pos() + 1 != hole.pos() and
            //  child + 1 == 2 * hole.pos() + 2 != hole.pos().
            // FIXME: 2 * hole.pos() + 1 or 2 * hole.pos() + 2 could overflow
            //  if T is a ZST
            child += unsafe { hole.get(child).key <= hole.get(child + 1).key } as usize;

            // if we are already in order, stop.
            // SAFETY: child is now either the old child or the old child+1
            //  We already proven that both are < self.len() and != hole.pos()
            if &hole.element().key >= unsafe { &hole.get(child).key } {
                return;
            }

            // SAFETY: same as above.
            unsafe { hole.move_to(child) };
            child = 2 * hole.pos() + 1;
        }

        // SAFETY: && short circuit, which means that in the
        //  second condition it's already true that child == end - 1 < self.len().
        if child == end - 1 && &hole.element().key < unsafe { &hole.get(child).key } {
            // SAFETY: child is already proven to be a valid index and
            //  child == 2 * hole.pos() + 1 != hole.pos().
            unsafe { hole.move_to(child) };
        }
    }

    /// # Safety
    ///
    /// The caller must guarantee that `pos < self.len()`.
    unsafe fn sift_down(&mut self, pos: usize) {
        let len = self.len();
        // SAFETY: pos < len is guaranteed by the caller and
        //  obviously len = self.len() <= self.len().
        unsafe { self.sift_down_range(pos, len) };
    }

    /// Take an element at `pos` and move it all the way down the heap,
    /// then sift it up to its position.
    ///
    /// Note: This is faster when the element is known to be large / should
    /// be closer to the bottom.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `pos < self.len()`.
    unsafe fn sift_down_to_bottom(&mut self, mut pos: usize) {
        let end = self.len();
        let start = pos;

        // SAFETY: The caller guarantees that pos < self.len().
        let mut hole = unsafe { Hole::new(&mut self.data, pos) };
        let mut child = 2 * hole.pos() + 1;

        // Loop invariant: child == 2 * hole.pos() + 1.
        while child <= end.saturating_sub(2) {
            // SAFETY: child < end - 1 < self.len() and
            //  child + 1 < end <= self.len(), so they're valid indexes.
            //  child == 2 * hole.pos() + 1 != hole.pos() and
            //  child + 1 == 2 * hole.pos() + 2 != hole.pos().
            // FIXME: 2 * hole.pos() + 1 or 2 * hole.pos() + 2 could overflow
            //  if T is a ZST
            child += unsafe { hole.get(child).key <= hole.get(child + 1).key } as usize;

            // SAFETY: Same as above
            unsafe { hole.move_to(child) };
            child = 2 * hole.pos() + 1;
        }

        if child == end - 1 {
            // SAFETY: child == end - 1 < self.len(), so it's a valid index
            //  and child == 2 * hole.pos() + 1 != hole.pos().
            unsafe { hole.move_to(child) };
        }
        pos = hole.pos();
        drop(hole);

        // SAFETY: pos is the position in the hole and was already proven
        //  to be a valid index.
        unsafe { self.sift_up(start, pos) };
    }

    /// Rebuild assuming data[0..start] is still a proper heap.
    fn rebuild_tail(&mut self, start: usize) {
        if start == self.len() {
            return;
        }

        let tail_len = self.len() - start;

        #[inline(always)]
        fn log2_fast(x: usize) -> usize {
            (usize::BITS - x.leading_zeros() - 1) as usize
        }

        // `rebuild` takes O(self.len()) operations
        // and about 2 * self.len() comparisons in the worst case
        // while repeating `sift_up` takes O(tail_len * log(start)) operations
        // and about 1 * tail_len * log_2(start) comparisons in the worst case,
        // assuming start >= tail_len. For larger heaps, the crossover point
        // no longer follows this reasoning and was determined empirically.
        let better_to_rebuild = if start < tail_len {
            true
        } else if self.len() <= 2048 {
            2 * self.len() < tail_len * log2_fast(start)
        } else {
            2 * self.len() < tail_len * 11
        };

        if better_to_rebuild {
            self.rebuild();
        } else {
            for i in start..self.len() {
                // SAFETY: The index `i` is always less than self.len().
                unsafe { self.sift_up(0, i) };
            }
        }
    }

    fn rebuild(&mut self) {
        let mut n = self.len() / 2;
        while n > 0 {
            n -= 1;
            // SAFETY: n starts from self.len() / 2 and goes down to 0.
            //  The only case when !(n < self.len()) is if
            //  self.len() == 0, but it's ruled out by the loop condition.
            unsafe { self.sift_down(n) };
        }
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    ///
    /// let mut a = KeyValueHeap::from([-10, 1, 2, 3, 3]);
    /// let mut b = KeyValueHeap::from([-20, 5, 43]);
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.into_sorted_vec(), [-20, -10, 1, 2, 3, 3, 5, 43]);
    /// assert!(b.is_empty());
    /// ```

    pub fn append(&mut self, other: &mut Self) {
        if self.len() < other.len() {
            swap(self, other);
        }

        let start = self.data.len();

        self.data.append(&mut other.data);

        self.rebuild_tail(start);
    }

    /// Clears the binary heap, returning an iterator over the removed elements
    /// in heap order. If the iterator is dropped before being fully consumed,
    /// it drops the remaining elements in heap order.
    ///
    /// The returned iterator keeps a mutable borrow on the heap to optimize
    /// its implementation.
    ///
    /// Note:
    /// * `.drain_sorted()` is *O*(*n* \* log(*n*)); much slower than `.drain()`.
    ///   You should use the latter for most cases.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(binary_heap_drain_sorted)]
    /// use kv_heap::KeyValueHeap;
    ///
    /// let mut heap = KeyValueHeap::from([1, 2, 3, 4, 5]);
    /// assert_eq!(heap.len(), 5);
    ///
    /// drop(heap.drain_sorted()); // removes all elements in heap order
    /// assert_eq!(heap.len(), 0);
    /// ```
    #[inline]
    pub fn drain_sorted(&mut self) -> DrainSorted<'_, K, V> {
        DrainSorted { inner: self }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns
    /// `false`. The elements are visited in unsorted (and unspecified) order.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    ///
    /// let mut heap = KeyValueHeap::from([-10, -5, 1, 2, 4, 13]);
    ///
    /// heap.retain(|_key, value| value % 2 == 0); // only keep even numbers
    ///
    /// assert_eq!(heap.into_sorted_vec(), [-10, 2, 4])
    /// ```

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        let mut first_removed = self.len();
        let mut i = 0;
        self.data.retain(|e| {
            let keep = f(&e.key, &e.value);
            if !keep && i < first_removed {
                first_removed = i;
            }
            i += 1;
            keep
        });
        // data[0..first_removed] is untouched, so we only need to rebuild the tail:
        self.rebuild_tail(first_removed);
    }
}

impl<K, V> KeyValueHeap<K, V> {
    /// Returns an iterator visiting all values in the underlying vector, in
    /// arbitrary order.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let heap = KeyValueHeap::from([1, 2, 3, 4]);
    ///
    /// // Print 1, 2, 3, 4 in arbitrary order
    /// for x in heap.iter() {
    ///     println!("{x}");
    /// }
    /// ```

    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            iter: self.data.iter(),
        }
    }

    /// Returns an iterator which retrieves elements in heap order.
    /// This method consumes the original heap.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let heap = KeyValueHeap::from([1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(heap.into_iter_sorted().take(2).collect::<Vec<_>>(), [5, 4]);
    /// ```
    pub fn into_iter_sorted(self) -> IntoIterSorted<K, V> {
        IntoIterSorted { inner: self }
    }

    /// Returns the greatest item in the binary heap, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::{KeyValueHeap, HeapNode};
    /// let mut heap = KeyValueHeap::new();
    /// assert_eq!(heap.peek(), None);
    ///
    /// heap.push(1, "Bob");
    /// heap.push(5, "Alice");
    /// heap.push(2, "Eve");
    /// assert_eq!(heap.peek(), Some(&HeapNode { key: 5, value: "Alice" }));
    ///
    /// ```
    ///
    /// # Time complexity
    ///
    /// Cost is *O*(1) in the worst case.
    #[must_use]

    pub fn peek(&self) -> Option<&HeapNode<K, V>> {
        self.data.get(0)
    }

    /// Returns the number of elements the binary heap can hold without reallocating.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let mut heap = KeyValueHeap::with_capacity(100);
    /// assert!(heap.capacity() >= 100);
    /// heap.push(4, "Bob");
    /// ```
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Reserves the minimum capacity for exactly `additional` more elements to be inserted in the
    /// given `KeyValueHeap`. Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore
    /// capacity can not be relied upon to be precisely minimal. Prefer [`reserve`] if future
    /// insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let mut heap = KeyValueHeap::new();
    /// heap.reserve_exact(100);
    /// assert!(heap.capacity() >= 100);
    /// heap.push(4, "Alice");
    /// ```
    ///
    /// [`reserve`]: KeyValueHeap::reserve

    pub fn reserve_exact(&mut self, additional: usize) {
        self.data.reserve_exact(additional);
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the
    /// `KeyValueHeap`. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let mut heap = KeyValueHeap::new();
    /// heap.reserve(100);
    /// assert!(heap.capacity() >= 100);
    /// heap.push(4, "Steven");
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Tries to reserve the minimum capacity for exactly `additional`
    /// elements to be inserted in the given `KeyValueHeap<T>`. After calling
    /// `try_reserve_exact`, capacity will be greater than or equal to
    /// `self.len() + additional` if it returns `Ok(())`.
    /// Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer [`try_reserve`] if future insertions are expected.
    ///
    /// [`try_reserve`]: KeyValueHeap::try_reserve
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(try_reserve_2)]
    /// use kv_heap::KeyValueHeap;
    /// use std::collections::TryReserveError;
    ///
    /// fn find_max_slow(data: &[u32]) -> Result<Option<u32>, TryReserveError> {
    ///     let mut heap = KeyValueHeap::new();
    ///
    ///     // Pre-reserve the memory, exiting if we can't
    ///     heap.try_reserve_exact(data.len())?;
    ///
    ///     // Now we know this can't OOM in the middle of our complex work
    ///     heap.extend(data.iter());
    ///
    ///     Ok(heap.pop().map(|e| *e.value))
    /// }
    /// # find_max_slow(&[1, 2, 3]).expect("why is the test harness OOMing on 12 bytes?");
    /// ```
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.data.try_reserve_exact(additional)
    }

    /// Tries to reserve capacity for at least `additional` more elements to be inserted
    /// in the given `KeyValueHeap<T>`. The collection may reserve more space to avoid
    /// frequent reallocations. After calling `try_reserve`, capacity will be
    /// greater than or equal to `self.len() + additional`. Does nothing if
    /// capacity is already sufficient.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(try_reserve_2)]
    /// use kv_heap::KeyValueHeap;
    /// use std::collections::TryReserveError;
    ///
    /// fn find_max_slow(data: &[u32]) -> Result<Option<u32>, TryReserveError> {
    ///     let mut heap = KeyValueHeap::new();
    ///
    ///     // Pre-reserve the memory, exiting if we can't
    ///     heap.try_reserve(data.len())?;
    ///
    ///     // Now we know this can't OOM in the middle of our complex work
    ///     heap.extend(data.iter());
    ///
    ///     Ok(heap.pop().map(|e| *e.value))
    /// }
    /// # find_max_slow(&[1, 2, 3]).expect("why is the test harness OOMing on 12 bytes?");
    /// ```
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.data.try_reserve(additional)
    }

    /// Discards as much additional capacity as possible.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let mut heap: KeyValueHeap<i32, &str> = KeyValueHeap::with_capacity(100);
    ///
    /// assert!(heap.capacity() >= 100);
    /// heap.shrink_to_fit();
    /// assert!(heap.capacity() == 0);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// Discards capacity with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length
    /// and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    ///
    /// # Examples
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let mut heap: KeyValueHeap<i32, &str> = KeyValueHeap::with_capacity(100);
    ///
    /// assert!(heap.capacity() >= 100);
    /// heap.shrink_to(10);
    /// assert!(heap.capacity() >= 10);
    /// ```
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.data.shrink_to(min_capacity)
    }

    /// Consumes the `KeyValueHeap` and returns the underlying vector
    /// in arbitrary order.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let heap = KeyValueHeap::from([1, 2, 3, 4, 5, 6, 7]);
    /// let vec = heap.into_vec();
    ///
    /// // Will print in some order
    /// for x in vec {
    ///     println!("{x}");
    /// }
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]

    pub fn into_vec(self) -> Vec<V> {
        self.into()
    }

    /// Returns the length of the binary heap.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let heap = KeyValueHeap::from([1, 3]);
    ///
    /// assert_eq!(heap.len(), 2);
    /// ```
    #[must_use]

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Checks if the binary heap is empty.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let mut heap = KeyValueHeap::new();
    ///
    /// assert!(heap.is_empty());
    ///
    /// heap.push(3, "Bob");
    /// heap.push(5, "Alice");
    /// heap.push(1, "Eve");
    ///
    /// assert!(!heap.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the binary heap, returning an iterator over the removed elements
    /// in arbitrary order. If the iterator is dropped before being fully
    /// consumed, it drops the remaining elements in arbitrary order.
    ///
    /// The returned iterator keeps a mutable borrow on the heap to optimize
    /// its implementation.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let mut heap = KeyValueHeap::from([1, 3]);
    ///
    /// assert!(!heap.is_empty());
    ///
    /// for x in heap.drain() {
    ///     println!("key: {} value: {}", x.key, x.value);
    /// }
    ///
    /// assert!(heap.is_empty());
    /// ```
    #[inline]

    pub fn drain(&mut self) -> Drain<'_, HeapNode<K, V>> {
        Drain {
            iter: self.data.drain(..),
        }
    }

    /// Drops all items from the binary heap.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use kv_heap::KeyValueHeap;
    /// let mut heap = KeyValueHeap::from([1, 3]);
    ///
    /// assert!(!heap.is_empty());
    ///
    /// heap.clear();
    ///
    /// assert!(heap.is_empty());
    /// ```

    pub fn clear(&mut self) {
        self.drain();
    }
}

impl<K: PartialEq, V> PartialEq for KeyValueHeap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for (lhs, rhs) in self.data.iter().zip(&other.data) {
            if *lhs != *rhs {
                return false;
            }
        }

        true
    }
}

impl<K: Eq, V> Eq for KeyValueHeap<K, V> {}

impl<K: Ord, V> PartialOrd for KeyValueHeap<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord, V> Ord for KeyValueHeap<K, V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.data.cmp(&other.data)
    }
}

/// Hole represents a hole in a slice i.e., an index without valid value
/// (because it was moved from or duplicated).
/// In drop, `Hole` will restore the slice by filling the hole
/// position with the value that was originally removed.
struct Hole<'a, T: 'a> {
    data: &'a mut [T],
    elt: ManuallyDrop<T>,
    pos: usize,
}

impl<'a, T> Hole<'a, T> {
    /// Create a new `Hole` at index `pos`.
    ///
    /// Unsafe because pos must be within the data slice.
    #[inline]
    #[allow(unused_unsafe)]
    unsafe fn new(data: &'a mut [T], pos: usize) -> Self {
        debug_assert!(pos < data.len());
        // SAFE: pos should be inside the slice
        let elt = unsafe { ptr::read(data.get_unchecked(pos)) };
        Hole {
            data,
            elt: ManuallyDrop::new(elt),
            pos,
        }
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }

    /// Returns a reference to the element removed.
    #[inline]
    fn element(&self) -> &T {
        &self.elt
    }

    /// Returns a reference to the element at `index`.
    ///
    /// Unsafe because index must be within the data slice and not equal to pos.
    #[inline]
    #[allow(unused_unsafe)]
    unsafe fn get(&self, index: usize) -> &T {
        debug_assert!(index != self.pos);
        debug_assert!(index < self.data.len());
        unsafe { self.data.get_unchecked(index) }
    }

    /// Move hole to new location
    ///
    /// Unsafe because index must be within the data slice and not equal to pos.
    #[inline]
    #[allow(unused_unsafe)]
    unsafe fn move_to(&mut self, index: usize) {
        debug_assert!(index != self.pos);
        debug_assert!(index < self.data.len());
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let index_ptr: *const _ = ptr.add(index);
            let hole_ptr = ptr.add(self.pos);
            ptr::copy_nonoverlapping(index_ptr, hole_ptr, 1);
        }
        self.pos = index;
    }
}

impl<T> Drop for Hole<'_, T> {
    #[inline]
    fn drop(&mut self) {
        // fill the hole again
        unsafe {
            let pos = self.pos;
            ptr::copy_nonoverlapping(&*self.elt, self.data.get_unchecked_mut(pos), 1);
        }
    }
}

/// An iterator over the elements of a `KeyValueHeap`.
///
/// This `struct` is created by [`KeyValueHeap::iter()`]. See its
/// documentation for more.
///
/// [`iter`]: KeyValueHeap::iter
#[must_use = "iterators are lazy and do nothing unless consumed"]

pub struct Iter<'a, K: 'a, V: 'a> {
    iter: slice::Iter<'a, HeapNode<K, V>>,
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Iter<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Iter").field(&self.iter.as_slice()).finish()
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`

impl<K, V> Clone for Iter<'_, K, V> {
    fn clone(&self) -> Self {
        Iter {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = &'a V;

    #[inline]
    fn next(&mut self) -> Option<&'a V> {
        self.iter.next().map(|node| &node.value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn last(self) -> Option<&'a V> {
        self.iter.last().map(|node| &node.value)
    }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a V> {
        self.iter.next_back().map(|node| &node.value)
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {
    // SNAP: is_empty (unstable)
}

impl<K, V> FusedIterator for Iter<'_, K, V> {}

/// An owning iterator over the elements of a `KeyValueHeap`.
///
/// This `struct` is created by [`KeyValueHeap::into_iter()`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: KeyValueHeap::into_iter
/// [`IntoIterator`]: core::iter::IntoIterator

#[derive(Clone)]
pub struct IntoIter<V> {
    iter: vec::IntoIter<V>,
}

impl<V: fmt::Debug> fmt::Debug for IntoIter<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter")
            .field(&self.iter.as_slice())
            .finish()
    }
}

impl<V> Iterator for IntoIter<V> {
    type Item = V;

    #[inline]
    fn next(&mut self) -> Option<V> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<V> DoubleEndedIterator for IntoIter<V> {
    #[inline]
    fn next_back(&mut self) -> Option<V> {
        self.iter.next_back()
    }
}

impl<V> ExactSizeIterator for IntoIter<V> {
    // SNAP: is_empty (unstable)
}

impl<V> FusedIterator for IntoIter<V> {}

// In addition to the SAFETY invariants of the following three unsafe traits
// also refer to the vec::in_place_collect module documentation to get an overview

#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct IntoIterSorted<K, V> {
    inner: KeyValueHeap<K, V>,
}

impl<K: Ord, V> Iterator for IntoIterSorted<K, V> {
    type Item = V;

    #[inline]
    fn next(&mut self) -> Option<V> {
        self.inner.pop().map(|node| node.value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.inner.len();
        (exact, Some(exact))
    }
}

impl<K: Ord, V> ExactSizeIterator for IntoIterSorted<K, V> {}

impl<K: Ord, V> FusedIterator for IntoIterSorted<K, V> {}

/// A draining iterator over the elements of a `KeyValueHeap`.
///
/// This `struct` is created by [`KeyValueHeap::drain()`]. See its
/// documentation for more.
///
/// [`drain`]: KeyValueHeap::drain

#[derive(Debug)]
pub struct Drain<'a, V: 'a> {
    iter: vec::Drain<'a, V>,
}

impl<V> Iterator for Drain<'_, V> {
    type Item = V;

    #[inline]
    fn next(&mut self) -> Option<V> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<V> DoubleEndedIterator for Drain<'_, V> {
    #[inline]
    fn next_back(&mut self) -> Option<V> {
        self.iter.next_back()
    }
}

impl<V> ExactSizeIterator for Drain<'_, V> {
    // SNAP: is_empty (unstable)
}

impl<V> FusedIterator for Drain<'_, V> {}

/// A draining iterator over the elements of a `KeyValueHeap`.
///
/// This `struct` is created by [`KeyValueHeap::drain_sorted()`]. See its
/// documentation for more.
///
/// [`drain_sorted`]: KeyValueHeap::drain_sorted

#[derive(Debug)]
pub struct DrainSorted<'a, K: Ord, V> {
    inner: &'a mut KeyValueHeap<K, V>,
}

impl<'a, K: Ord, V> Drop for DrainSorted<'a, K, V> {
    /// Removes heap elements in heap order.
    fn drop(&mut self) {
        struct DropGuard<'r, 'a, K: Ord, V>(&'r mut DrainSorted<'a, K, V>);

        impl<'r, 'a, K: Ord, V> Drop for DropGuard<'r, 'a, K, V> {
            fn drop(&mut self) {
                while self.0.inner.pop().is_some() {}
            }
        }

        while let Some(item) = self.inner.pop() {
            let guard = DropGuard(self);
            drop(item);
            mem::forget(guard);
        }
    }
}

impl<K: Ord, V> Iterator for DrainSorted<'_, K, V> {
    type Item = V;

    #[inline]
    fn next(&mut self) -> Option<V> {
        self.inner.pop().map(|node| node.value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.inner.len();
        (exact, Some(exact))
    }
}

impl<K: Ord, V> ExactSizeIterator for DrainSorted<'_, K, V> {}

impl<K: Ord, V> FusedIterator for DrainSorted<'_, K, V> {}

impl<K: Clone + Ord> From<Vec<K>> for KeyValueHeap<K, K> {
    /// Converts a `Vec<T>` into a `KeyValueHeap<T>`.
    ///
    /// This conversion happens in-place, and has *O*(*n*) time complexity.
    fn from(vec: Vec<K>) -> KeyValueHeap<K, K> {
        let mut heap = KeyValueHeap {
            data: vec
                .into_iter()
                .map(|key| HeapNode {
                    value: key.clone(),
                    key,
                })
                .collect(),
        };
        heap.rebuild();
        heap
    }
}

impl<K: Ord, V> From<Vec<HeapNode<K, V>>> for KeyValueHeap<K, V> {
    /// Converts a `Vec<T>` into a `KeyValueHeap<T>`.
    ///
    /// This conversion happens in-place, and has *O*(*n*) time complexity.
    fn from(vec: Vec<HeapNode<K, V>>) -> KeyValueHeap<K, V> {
        let mut heap = KeyValueHeap { data: vec };
        heap.rebuild();
        heap
    }
}

impl<K: Ord, V> From<Vec<(K, V)>> for KeyValueHeap<K, V> {
    /// Converts a `Vec<T>` into a `KeyValueHeap<T>`.
    ///
    /// This conversion happens in-place, and has *O*(*n*) time complexity.
    fn from(vec: Vec<(K, V)>) -> KeyValueHeap<K, V> {
        let mut heap = KeyValueHeap {
            data: vec
                .into_iter()
                .map(|(key, value)| HeapNode { key, value })
                .collect(),
        };
        heap.rebuild();
        heap
    }
}

impl<K: Ord, V, const N: usize> From<[(K, V); N]> for KeyValueHeap<K, V> {
    /// ```
    /// use kv_heap::KeyValueHeap;
    ///
    /// let mut h1 = KeyValueHeap::from([1, 4, 2, 3]);
    /// let mut h2: KeyValueHeap<_, _> = [1, 4, 2, 3].into();
    /// while let Some((a, b)) = h1.pop().zip(h2.pop()) {
    ///     assert_eq!(a, b);
    /// }
    /// ```
    fn from(arr: [(K, V); N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<K: Clone + Ord, const N: usize> From<[K; N]> for KeyValueHeap<K, K> {
    /// ```
    /// use kv_heap::KeyValueHeap;
    ///
    /// let mut h1 = KeyValueHeap::from([1, 4, 2, 3]);
    /// let mut h2: KeyValueHeap<_, _> = [1, 4, 2, 3].into();
    /// while let Some((a, b)) = h1.pop().zip(h2.pop()) {
    ///     assert_eq!(a, b);
    /// }
    /// ```
    fn from(arr: [K; N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<K, V> From<KeyValueHeap<K, V>> for Vec<HeapNode<K, V>> {
    /// Converts a `KeyValueHeap<T>` into a `Vec<T>`.
    ///
    /// This conversion requires no data movement or allocation, and has
    /// constant time complexity.
    fn from(heap: KeyValueHeap<K, V>) -> Vec<HeapNode<K, V>> {
        heap.data
    }
}

impl<K, V> From<KeyValueHeap<K, V>> for Vec<V> {
    /// Converts a `KeyValueHeap<T>` into a `Vec<T>`.
    ///
    /// This conversion requires no data movement or allocation, and has
    /// constant time complexity.
    fn from(heap: KeyValueHeap<K, V>) -> Vec<V> {
        heap.data.into_iter().map(|node| node.value).collect()
    }
}

impl<K: Ord + Clone> FromIterator<K> for KeyValueHeap<K, K> {
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> KeyValueHeap<K, K> {
        KeyValueHeap::from(iter.into_iter().map(|v| (v.clone(), v)).collect::<Vec<_>>())
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for KeyValueHeap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> KeyValueHeap<K, V> {
        KeyValueHeap::from(iter.into_iter().collect::<Vec<_>>())
    }
}

impl<K: Ord, V> FromIterator<HeapNode<K, V>> for KeyValueHeap<K, V> {
    fn from_iter<I: IntoIterator<Item = HeapNode<K, V>>>(iter: I) -> KeyValueHeap<K, V> {
        KeyValueHeap::from(iter.into_iter().collect::<Vec<_>>())
    }
}

// impl<K, T> IntoIterator for KeyValueHeap<K, T> {
//     type Item = T;
//     type IntoIter = IntoIter<Self::Item>;

//     /// Creates a consuming iterator, that is, one that moves each value out of
//     /// the binary heap in arbitrary order. The binary heap cannot be used
//     /// after calling this.
//     ///
//     /// # Examples
//     ///
//     /// Basic usage:
//     ///
//     /// ```
//     /// use kv_heap::KeyValueHeap;
//     /// let heap = KeyValueHeap::from([1, 2, 3, 4]);
//     ///
//     /// // Print 1, 2, 3, 4 in arbitrary order
//     /// for x in heap.into_iter() {
//     ///     // x has type i32, not &i32
//     ///     println!("{x}");
//     /// }
//     /// ```
//     fn into_iter(self) -> IntoIter<T> {
//         IntoIter {
//             iter: self.data.into_iter().map(|node| node.1).collect(),
//         }
//     }
// }

impl<'a, K, V> IntoIterator for &'a KeyValueHeap<K, V> {
    type Item = &'a V;
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<K: Clone + Ord> Extend<K> for KeyValueHeap<K, K> {
    #[inline]
    fn extend<I: IntoIterator<Item = K>>(&mut self, iter: I) {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();

        self.reserve(lower);

        iterator.for_each(move |v| self.push(v.clone(), v));
    }
}

impl<K: Ord, V> Extend<(K, V)> for KeyValueHeap<K, V> {
    #[inline]
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();

        self.reserve(lower);

        iterator.for_each(move |(key, value)| self.push(key, value));
    }
}

// Node iteration

// impl<K: Ord, V> IntoIterator for KeyValueHeap<K, V> {
//     type Item = HeapNode<K, V>;
//     type IntoIter = <Vec<Self::Item> as IntoIterator>::IntoIter;

//     fn into_iter(self) -> Self::IntoIter {
//         self.data.into_iter()
//     }
// }

// impl<K: Ord, V> Extend<HeapNode<K, V>> for KeyValueHeap<K, V> {
//     #[inline]
//     fn extend<I: IntoIterator<Item = HeapNode<K, V>>>(&mut self, iter: I) {
//         let iterator = iter.into_iter();
//         let (lower, _) = iterator.size_hint();

//         self.reserve(lower);

//         iterator.for_each(move |HeapNode(key, value)| self.push(key, value));
//     }
// }
