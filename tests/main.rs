#![feature(box_syntax)]

use kv_heap::Drain;
use kv_heap::KeyValueHeap;
// use std::panic::{catch_unwind, AssertUnwindSafe};
// use std::sync::atomic::{AtomicU32, Ordering};

#[test]
fn test_iterator() {
    let data = vec![5, 9, 3];
    let iterout = [9, 5, 3];
    let heap = KeyValueHeap::from(data);
    let mut i = 0;
    for el in &heap {
        assert_eq!(*el, iterout[i]);
        i += 1;
    }
}

#[test]
fn test_iter_rev_cloned_collect() {
    let data = vec![5, 9, 3];
    let iterout = vec![3, 5, 9];
    let pq = KeyValueHeap::from(data);

    let v: Vec<_> = pq.iter().rev().cloned().collect();
    assert_eq!(v, iterout);
}

// #[test]
// fn test_into_iter_collect() {
//     let data = vec![5, 9, 3];
//     let iterout = vec![9, 5, 3];
//     let pq = BinaryHeap::from(data);

//     let v: Vec<_> = pq.into_iter().collect();
//     assert_eq!(v, iterout);
// }

#[test]
fn test_into_iter_size_hint() {
    let data = vec![5, 9];
    let pq = KeyValueHeap::from(data);

    let mut it = pq.into_iter();

    assert_eq!(it.size_hint(), (2, Some(2)));
    assert_eq!(it.next(), Some(&9));

    assert_eq!(it.size_hint(), (1, Some(1)));
    assert_eq!(it.next(), Some(&5));

    assert_eq!(it.size_hint(), (0, Some(0)));
    assert_eq!(it.next(), None);
}

// #[test]
// fn test_into_iter_rev_collect() {
//     let data = vec![5, 9, 3];
//     let iterout = vec![3, 5, 9];
//     let pq = BinaryHeap::from(data);

//     let v: Vec<_> = pq.into_iter().rev().collect();
//     assert_eq!(v, iterout);
// }

#[test]
fn test_into_iter_sorted_collect() {
    let heap = KeyValueHeap::from(vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1]);
    let it = heap.into_iter_sorted();
    let sorted = it.collect::<Vec<_>>();
    assert_eq!(sorted, vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 1, 1, 0]);
}

#[test]
fn test_drain_sorted_collect() {
    let mut heap = KeyValueHeap::from(vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1]);
    let it = heap.drain_sorted();
    let sorted = it.collect::<Vec<_>>();
    assert_eq!(sorted, vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 1, 1, 0]);
}

fn check_exact_size_iterator<I: ExactSizeIterator>(len: usize, it: I) {
    let mut it = it;

    for i in 0..it.len() {
        let (lower, upper) = it.size_hint();
        assert_eq!(Some(lower), upper);
        assert_eq!(lower, len - i);
        assert_eq!(it.len(), len - i);
        it.next();
    }
    assert_eq!(it.len(), 0);
    // assert!(it.is_empty());
}

#[test]
fn test_exact_size_iterator() {
    let heap = KeyValueHeap::from(vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1]);
    check_exact_size_iterator(heap.len(), heap.iter());
    check_exact_size_iterator(heap.len(), heap.clone().into_iter());
    check_exact_size_iterator(heap.len(), heap.clone().into_iter_sorted());
    check_exact_size_iterator(heap.len(), heap.clone().drain());
    check_exact_size_iterator(heap.len(), heap.clone().drain_sorted());
}

#[test]
fn test_peek_and_pop() {
    let data = vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
    let mut sorted = data.clone();
    sorted.sort();
    let mut heap = KeyValueHeap::from(data);
    while !heap.is_empty() {
        assert_eq!(heap.peek().unwrap().value, *sorted.last().unwrap());
        assert_eq!(heap.pop().unwrap().value, sorted.pop().unwrap());
    }
}

// #[test]
// fn test_peek_mut() {
//     let data = vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
//     let mut heap = BinaryHeap::from(data);
//     assert_eq!(heap.peek(), Some(&10));
//     {
//         let mut top = heap.peek_mut().unwrap();
//         *top -= 2;
//     }
//     assert_eq!(heap.peek(), Some(&9));
// }

// #[test]
// fn test_peek_mut_pop() {
//     let data = vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
//     let mut heap = BinaryHeap::from(data);
//     assert_eq!(heap.peek(), Some(&10));
//     {
//         let mut top = heap.peek_mut().unwrap();
//         *top -= 2;
//         assert_eq!(PeekMut::pop(top), 8);
//     }
//     assert_eq!(heap.peek(), Some(&9));
// }

#[test]
fn test_push() {
    let mut heap = KeyValueHeap::from(vec![2, 4, 9]);
    assert_eq!(heap.len(), 3);
    assert!(heap.peek().unwrap().value == 9);
    heap.push(11, 11);
    assert_eq!(heap.len(), 4);
    assert!(heap.peek().unwrap().value == 11);
    heap.push(5, 5);
    assert_eq!(heap.len(), 5);
    assert!(heap.peek().unwrap().value == 11);
    heap.push(27, 27);
    assert_eq!(heap.len(), 6);
    assert!(heap.peek().unwrap().value == 27);
    heap.push(3, 3);
    assert_eq!(heap.len(), 7);
    assert!(heap.peek().unwrap().value == 27);
    heap.push(103, 103);
    assert_eq!(heap.len(), 8);
    assert!(heap.peek().unwrap().value == 103);
}

#[test]
fn test_push_unique() {
    let mut heap = KeyValueHeap::from(vec![box 2, box 4, box 9]);
    assert_eq!(heap.len(), 3);
    assert!(*heap.peek().unwrap().value == 9);
    heap.push(box 11, box 11);
    assert_eq!(heap.len(), 4);
    assert!(*heap.peek().unwrap().value == 11);
    heap.push(box 5, box 5);
    assert_eq!(heap.len(), 5);
    assert!(*heap.peek().unwrap().value == 11);
    heap.push(box 27, box 27);
    assert_eq!(heap.len(), 6);
    assert!(*heap.peek().unwrap().value == 27);
    heap.push(box 3, box 3);
    assert_eq!(heap.len(), 7);
    assert!(*heap.peek().unwrap().value == 27);
    heap.push(box 103, box 103);
    assert_eq!(heap.len(), 8);
    assert!(*heap.peek().unwrap().value == 103);
}

fn check_to_vec(mut data: Vec<i32>) {
    let heap = KeyValueHeap::from(data.clone());
    let mut v: Vec<i32> = heap.clone().into_vec();
    v.sort();
    data.sort();

    assert_eq!(v, data);
    assert_eq!(heap.into_sorted_vec(), data);
}

#[test]
fn test_to_vec() {
    check_to_vec(vec![]);
    check_to_vec(vec![5]);
    check_to_vec(vec![3, 2]);
    check_to_vec(vec![2, 3]);
    check_to_vec(vec![5, 1, 2]);
    check_to_vec(vec![1, 100, 2, 3]);
    check_to_vec(vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 0]);
    check_to_vec(vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1]);
    check_to_vec(vec![9, 11, 9, 9, 9, 9, 11, 2, 3, 4, 11, 9, 0, 0, 0, 0]);
    check_to_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    check_to_vec(vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    check_to_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 2]);
    check_to_vec(vec![5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1]);
}

// #[test]
// fn test_in_place_iterator_specialization() {
//     let src: Vec<usize> = vec![1, 2, 3];
//     let src_ptr = src.as_ptr();
//     let heap: BinaryHeap<_> = src.into_iter().map(std::convert::identity).collect();
//     let heap_ptr = heap.iter().next().unwrap() as *const usize;
//     assert_eq!(src_ptr, heap_ptr);
//     let sink: Vec<_> = heap.into_iter().map(std::convert::identity).collect();
//     let sink_ptr = sink.as_ptr();
//     assert_eq!(heap_ptr, sink_ptr);
// }

#[test]
fn test_empty_pop() {
    let mut heap = KeyValueHeap::<i32, ()>::new();
    assert!(heap.pop().is_none());
}

#[test]
fn test_empty_peek() {
    let empty = KeyValueHeap::<i32, ()>::new();
    assert!(empty.peek().is_none());
}

#[test]
fn test_empty_peek_mut() {
    let mut empty = KeyValueHeap::<i32, ()>::new();
    assert!(empty.peek_mut().is_none());
}

#[test]
fn test_from_iter() {
    let xs = vec![9, 8, 7, 6, 5, 4, 3, 2, 1];

    let mut q: KeyValueHeap<_, _> = xs.iter().rev().cloned().collect();

    for &x in &xs {
        assert_eq!(q.pop().unwrap().key, x);
    }
}

#[test]
fn test_drain() {
    let mut q: KeyValueHeap<_, _> = [9, 8, 7, 6, 5, 4, 3, 2, 1].iter().cloned().collect();

    assert_eq!(q.drain().take(5).count(), 5);

    assert!(q.is_empty());
}

#[test]
fn test_drain_sorted() {
    let mut q: KeyValueHeap<_, _> = [9, 8, 7, 6, 5, 4, 3, 2, 1].iter().cloned().collect();

    assert_eq!(
        q.drain_sorted().take(5).collect::<Vec<_>>(),
        vec![9, 8, 7, 6, 5]
    );

    assert!(q.is_empty());
}

// #[test]
// fn test_drain_sorted_leak() {
//     static DROPS: AtomicU32 = AtomicU32::new(0);

//     #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
//     struct D(u32, bool);

//     impl Drop for D {
//         fn drop(&mut self) {
//             DROPS.fetch_add(1, Ordering::SeqCst);

//             if self.1 {
//                 panic!("panic in `drop`");
//             }
//         }
//     }

//     let mut q = BinaryHeap::from(vec![
//         D(0, false),
//         D(1, false),
//         D(2, false),
//         D(3, true),
//         D(4, false),
//         D(5, false),
//     ]);

//     drop(q.drain_sorted());
//     //catch_unwind(AssertUnwindSafe(|| drop(q.drain_sorted()))).ok();

//     assert_eq!(DROPS.load(Ordering::SeqCst), 6);
// }

#[test]
fn test_increase_key() {
    let heap_a =
        KeyValueHeap::<usize, String>::new_with_mapping([1, 2, 3, 5, 7, 10], |i| format!("{}", i));
    let mut heap_b =
        KeyValueHeap::<usize, String>::new_with_mapping([1, 2, 3, 5, 7, 10], |i| format!("{}", i));

    assert!(heap_b.check_integrity());

    assert_eq!(heap_a, heap_b);

    heap_b.change_key("7".to_string(), 11);
    assert!(heap_b.check_integrity());

    assert_ne!(heap_a, heap_b);
    assert_eq!(
        heap_b.clone().into_sorted_vec(),
        ["1", "2", "3", "5", "10", "7"]
    );

    heap_b.change_key("2".to_string(), 9);
    assert!(heap_b.check_integrity());

    assert_ne!(heap_a, heap_b);
    assert_eq!(
        heap_b.clone().into_sorted_vec(),
        ["1", "3", "5", "2", "10", "7"]
    );

    heap_b.change_key("2".to_string(), 15);
    assert!(heap_b.check_integrity());

    assert_ne!(heap_a, heap_b);
    assert_eq!(
        heap_b.clone().into_sorted_vec(),
        ["1", "3", "5", "10", "7", "2"]
    );

    heap_b.change_key("2".to_string(), 0);
    assert!(heap_b.check_integrity());

    assert_ne!(heap_a, heap_b);
    assert_eq!(
        heap_b.clone().into_sorted_vec(),
        ["2", "1", "3", "5", "10", "7"]
    );
}

#[test]
fn test_extend_ref() {
    let mut a = KeyValueHeap::new();
    a.push(1, 1);
    a.push(2, 2);

    a.extend([3, 4, 5]);

    assert_eq!(a.len(), 5);
    assert_eq!(a.into_sorted_vec(), [1, 2, 3, 4, 5]);

    let mut a = KeyValueHeap::new();
    a.push(1, 1);
    a.push(2, 2);
    let mut b = KeyValueHeap::new();
    b.push(3, 3);
    b.push(4, 4);
    b.push(5, 5);

    // a.extend(&b);

    // assert_eq!(a.len(), 5);
    // assert_eq!(a.into_sorted_vec(), [1, 2, 3, 4, 5]);
}

#[test]
fn test_append() {
    let mut a = KeyValueHeap::from(vec![-10, 1, 2, 3, 3]);
    let mut b = KeyValueHeap::from(vec![-20, 5, 43]);

    a.append(&mut b);

    assert_eq!(a.into_sorted_vec(), [-20, -10, 1, 2, 3, 3, 5, 43]);
    assert!(b.is_empty());
}

#[test]
fn test_append_to_empty() {
    let mut a = KeyValueHeap::new();
    let mut b = KeyValueHeap::from(vec![-20, 5, 43]);

    a.append(&mut b);

    assert_eq!(a.into_sorted_vec(), [-20, 5, 43]);
    assert!(b.is_empty());
}

// #[test]
// fn test_extend_specialization() {
//     let mut a = BinaryHeap::from(vec![-10, 1, 2, 3, 3]);
//     let b = BinaryHeap::from(vec![-20, 5, 43]);

//     // TODO
//     // a.extend(b);

//     assert_eq!(a.into_sorted_vec(), [-20, -10, 1, 2, 3, 3, 5, 43]);
// }

#[allow(dead_code)]
fn assert_covariance() {
    fn drain<'new>(d: Drain<'static, &'static str>) -> Drain<'new, &'new str> {
        d
    }
}

#[test]
fn test_retain() {
    let mut a = KeyValueHeap::from(vec![100, 10, 50, 1, 2, 20, 30]);
    a.retain(|_, &x| x != 2);

    // Check that 20 moved into 10's place.
    assert_eq!(a.clone().into_vec(), [100, 20, 50, 1, 10, 30]);

    a.retain(|_, _| true);

    assert_eq!(a.clone().into_vec(), [100, 20, 50, 1, 10, 30]);

    a.retain(|_, &x| x < 50);

    assert_eq!(a.clone().into_vec(), [30, 20, 10, 1]);

    a.retain(|_, _| false);

    assert!(a.is_empty());
}
