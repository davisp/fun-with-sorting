//! Multi-median detection
//!
//! Given a slice of sorted slices, this algorithm will return a Vec<usize>
//! that contains the last index in the list that falls in the first half of
//! the combined sorted list of all input lists. I.e., for each element in the
//! returned Vec, all elements less than the corresponding element will be
//! in the first half of the combined sorted list.
//!
//! This algorithm can be used to parallelize merge sorts by showing which
//! elements of each list are in the first or second half of the combined sorted
//! list. With this, we can avoid the final single threaded merge step in a
//! parallel merge sort.

// A couple woefully incomplete descriptions of the base algorithm:
//
// https://notexponential.com/174/how-find-the-joint-median-sorted-lists-less-than-linear-time
// https://stackoverflow.com/questions/6182488/median-of-5-sorted-arrays

#[derive(Clone, Debug, Default)]
struct ListInfo {
    list_idx: usize,
    median_idx_min: usize,
    median_idx_max: usize,
}

impl ListInfo {
    fn new(list_idx: usize, list_len: usize) -> Self {
        assert!(list_len > 0);
        Self {
            list_idx,
            median_idx_min: 0,
            median_idx_max: list_len - 1,
        }
    }

    fn num_elems(&self) -> usize {
        if self.median_idx_max < self.median_idx_min {
            return 0;
        }

        self.median_idx_max - self.median_idx_min + 1
    }

    // Since we're relying on the generic Ord trait, we're unable to use the
    // standard median algorithm that averages the two middle values in even
    // length lists.
    //
    // Our approach to this issue requires us to use both of the possible values
    // in the even length list case. We handle this by sorting twice. First we
    // sort "to the left" which uses the left side of the middle pair to get
    // the smallest median. We then sort a second time "to the right" to get
    // the largest median.
    fn median_idx_left(&self) -> usize {
        assert!(self.median_idx_min <= self.median_idx_max);

        let diff = self.median_idx_max - self.median_idx_min + 1;
        let mid = if diff >= 2 && diff % 2 == 0 {
            (diff - 1) / 2
        } else {
            diff / 2
        };

        self.median_idx_min + mid
    }

    fn median_idx_right(&self) -> usize {
        assert!(self.median_idx_min <= self.median_idx_max);

        let diff = self.median_idx_max - self.median_idx_min + 1;
        let mid = if diff >= 2 && diff % 2 == 0 {
            (diff + 1) / 2
        } else {
            diff / 2
        };

        self.median_idx_min + mid
    }
}

// Find the median of a slice of sorted slices.
//
// This returns the list and element indices of the median value if one exists.
//
// If the input to this function is not a slice of sorted slices, the output is
// not guaranteed to be meaningful.
pub fn find<T>(lists: &[&[T]]) -> Option<(usize, usize)>
where
    T: Ord + std::fmt::Debug,
{
    // Extra bookkeeping to aid in making assertions on behavior.
    let mut elems_remaining: usize = lists.iter().map(|list| list.len()).sum();

    // Nothing to sort, nothing to report.
    if lists.is_empty() {
        return None;
    }

    let medians = lists
        .iter()
        .enumerate()
        // Filter after enumerate so that list indices are correct.
        .filter(|(_, list)| !list.is_empty())
        .map(|(idx, sublist)| ListInfo::new(idx, sublist.len()))
        .collect::<Vec<_>>();

    // Nothing to sort, nothing to report.
    if medians.is_empty() {
        return None;
    }

    // A single non-empty slice means the median is trivial.
    if medians.len() < 2 {
        return Some((medians[0].list_idx, medians[0].median_idx_right()));
    }

    // This algorithm is in two phases that are only slightly different to
    // account for some subtle edge cases around termination. In phase 1, we
    // throw away elements from either end of the sorted median lists including
    // the proposed median. This allows for lists which cannot contain the
    // median to be rejected for consideration as containing the actual
    // median. See below for a description of Phase 2.

    let mut medians = medians;

    // N.B., we only iterate in Phase 1 as long as we have *three* (3!) or
    // more slices to process. Once we have 2 or 1, Phase 2 begins.
    while medians.len() > 2 {
        assert!(elems_remaining > 0);

        // We're using a list of integers to sort here so that we don't have
        // an O(N) step finding the smallest median after the sort right step.
        let mut indices = (0..medians.len()).collect::<Vec<_>>();
        assert!(indices.len() > 1 && indices.len() == medians.len());

        // Sort medians left to find the smallest median
        indices.sort_by(|lhs, rhs| {
            let m1 = &medians[*lhs];
            let m2 = &medians[*rhs];

            lists[m1.list_idx][m1.median_idx_left()]
                .cmp(&lists[m2.list_idx][m2.median_idx_left()])
        });

        let smallest_idx = *indices.first().unwrap();

        // Sort medians right to find the largest median
        indices.sort_by(|lhs, rhs| {
            let m1 = &medians[*lhs];
            let m2 = &medians[*rhs];

            lists[m1.list_idx][m1.median_idx_right()]
                .cmp(&lists[m2.list_idx][m2.median_idx_right()])
        });

        let largest_idx = *indices.last().unwrap();

        // If the smallest median equals the largest median, then everything
        // in-between is also equal and we're done.
        let smallest = &medians[smallest_idx];
        let largest = &medians[largest_idx];
        if lists[medians[smallest_idx].list_idx][smallest.median_idx_left()]
            == lists[largest.list_idx][largest.median_idx_right()]
        {
            return Some((smallest.list_idx, smallest.median_idx_left()));
        }

        // Occasionally, we can run into the situation where the smallest and
        // largest indices refer to the same list. This happens in the case of
        // something like `[[a, d], [b], [c]]`. When this case occurs, we can
        // just discard the list and continue on.
        if largest_idx == smallest_idx {
            let elems_in_list = medians[smallest_idx].num_elems();
            elems_remaining -= elems_in_list;

            let smallest_list_idx = medians[smallest_idx].list_idx;

            medians = medians
                .into_iter()
                .filter(|m| m.list_idx != smallest_list_idx)
                .collect::<Vec<_>>();

            // We started with more than one list, we're only removing one
            // list. Therefore, we must have at least one list with one
            // element.
            assert!(elems_remaining >= 1);
            continue;
        }

        // We just handled this case so its no longer possible here.
        assert_ne!(smallest_idx, largest_idx);

        let smallest = &medians[smallest_idx];
        let largest = &medians[largest_idx];

        // Sanity checking
        assert!(smallest.num_elems() > 0);
        assert!(largest.num_elems() > 0);

        let small_incr =
            smallest.median_idx_left() - smallest.median_idx_min + 1;
        let large_incr =
            largest.median_idx_max - largest.median_idx_right() + 1;

        // We only ever throw away the same number of elements from both ends
        // of our sorted arrays.
        let incr = std::cmp::min(small_incr, large_incr);

        if incr > 0 {
            medians[smallest_idx].median_idx_min += incr;

            // This is awkward. If we're about to filter this slice out, we
            // will quite probably undeflow the median_idx_max value. In that
            // case we set median_idx_max to 0, and median_idx_min to 1 to
            // indicate it needs to be removed.
            if medians[largest_idx].median_idx_max >= incr {
                medians[largest_idx].median_idx_max -= incr;
            } else {
                medians[largest_idx].median_idx_min = 1;
                medians[largest_idx].median_idx_max = 0;
            }

            elems_remaining -= incr * 2;

            // Its possible that one or both of these slices were just
            // completely consumed so we have to filter them out if so.
            let smallest_list_idx = if medians[smallest_idx].num_elems() == 0 {
                Some(medians[smallest_idx].list_idx)
            } else {
                None
            };

            let largest_list_idx = if medians[largest_idx].num_elems() == 0 {
                Some(medians[largest_idx].list_idx)
            } else {
                None
            };

            medians = medians
                .into_iter()
                .filter(|m| {
                    let list_idx = Some(m.list_idx);
                    list_idx != smallest_list_idx
                        && list_idx != largest_list_idx
                })
                .collect::<Vec<_>>();

            // TODO: Remove this once I think this is all correct.
            let elems: usize =
                medians.iter().map(|list| list.num_elems()).sum();
            assert_eq!(elems, elems_remaining);
            assert!(elems_remaining > 0);

            continue;
        }

        // Should have continued after processing non-zero incr
        assert_eq!(incr, 0);

        // If only one of the lists is being removed, we need to adjust the
        // other list by one to account for the last element of the popped
        // list being removed from the search.
        if small_incr > 0 && large_incr == 0 {
            medians[smallest_idx].median_idx_min += 1;
        }

        if small_incr == 0 && large_incr > 0 {
            medians[largest_idx].median_idx_max -= 1;
        }

        // Account for both elements removed. It doesn't matter if only one
        // list is removed or not, we know that we've removed exactly two in
        // total.
        elems_remaining -= 2;

        // At this point, one or both of smallest and largest have been
        // exhausted and should be removed from the search for the median.

        let smallest_list_idx = if medians[smallest_idx].num_elems() == 1 {
            Some(smallest_idx)
        } else {
            None
        };

        let largest_list_idx = if medians[largest_idx].num_elems() == 1 {
            Some(largest_idx)
        } else {
            None
        };

        medians = medians
            .into_iter()
            .filter(|m| {
                let list_idx = Some(m.list_idx);
                list_idx != smallest_list_idx && list_idx != largest_list_idx
            })
            .collect::<Vec<_>>();

        // Here we're wasting some CPU cycles to ensure that we catch any bugs
        // as early as possible.
        //
        // TODO: Remove this once I think this is all correct.
        let elems: usize = medians.iter().map(|list| list.num_elems()).sum();

        assert_eq!(elems, elems_remaining);
        assert!(elems_remaining >= 1);
    }

    // Phase 2: At this point we have either 1 or 2 slices remaining, one of
    // which contains the actual median.

    // We are guaranteed to have exactly one or exactly two slices left.
    assert!(medians.len() == 1 || medians.len() == 2);

    // If we lucked out, we only have a single slice left and can calculate
    // the median directly.
    if medians.len() == 1 {
        let median = medians.last().unwrap();

        // Check our element count record keeping.
        assert_eq!(
            median.median_idx_max - median.median_idx_min + 1,
            elems_remaining
        );

        // N.B., we're taking the right median here as that's the element that will
        // be in the top sorted list when the last slice has an even length.
        return Some((median.list_idx, median.median_idx_right()));
    }

    // We have two slices. First, we run the trimming steps until of of the
    // two slices can not be reduced any further. We must have at least one
    // element remain in both lists for the final calculation.
    //
    // Remember, this is quite close to Phase 1, but with some fairly subtle
    // changes to edge conditions so that we don't remove either slice from
    // consideration.
    let (smallest, largest) = loop {
        assert!(elems_remaining > 0);

        // Setup our faux index pointers to find the smallest and largest
        // slices indices.
        let mut indices = (0..medians.len()).collect::<Vec<_>>();
        assert!(indices.len() > 1 && indices.len() == medians.len());

        indices.sort_by(|lhs, rhs| {
            let m1 = &medians[*lhs];
            let m2 = &medians[*rhs];

            lists[m1.list_idx][m1.median_idx_left()]
                .cmp(&lists[m2.list_idx][m2.median_idx_left()])
        });

        let smallest_idx = *indices.first().unwrap();

        indices.sort_by(|lhs, rhs| {
            let m1 = &medians[*lhs];
            let m2 = &medians[*rhs];

            lists[m1.list_idx][m1.median_idx_right()]
                .cmp(&lists[m2.list_idx][m2.median_idx_right()])
        });

        let largest_idx = *indices.last().unwrap();

        // We only have two possible indices at this point.
        assert!(smallest_idx < 2 && largest_idx < 2);

        // Similar to before, if one of the slices encapsulates the other, we
        // know that the median of the inner slice is the final median.
        if largest_idx == smallest_idx {
            let elems_in_list = medians[smallest_idx].num_elems();

            elems_remaining -= elems_in_list;
            let smallest_list_idx = medians[smallest_idx].list_idx;

            medians = medians
                .into_iter()
                .filter(|m| m.list_idx != smallest_list_idx)
                .collect::<Vec<_>>();

            // However, different than before, the median of the inner slice
            // is guaranteed to be the median overall. Thus we can return it
            // here.

            assert_eq!(medians.len(), 1);
            assert_eq!(medians[0].num_elems(), elems_remaining);

            return Some((medians[0].list_idx, medians[0].median_idx_right()));
        }

        // We just handled this case so its no longer possible here.
        assert_ne!(smallest_idx, largest_idx);

        let smallest = &medians[smallest_idx];
        let largest = &medians[largest_idx];

        let small_incr = smallest.median_idx_left() - smallest.median_idx_min;
        let large_incr = largest.median_idx_max - largest.median_idx_right();

        // We only ever throw away the same number of elements from both ends
        // of our sorted arrays.
        let incr = std::cmp::min(small_incr, large_incr);

        if incr > 0 {
            medians[smallest_idx].median_idx_min += incr;
            medians[largest_idx].median_idx_max -= incr;
            elems_remaining -= incr * 2;

            assert!(elems_remaining > 0);

            continue;
        }

        // We are no longer able to make progress my successively trimming
        // elements from both slices.
        break (smallest, largest);
    };

    // Bookkeeping, make sure we still have a valid number of elements left.
    assert_eq!(smallest.num_elems() + largest.num_elems(), elems_remaining);

    // Given that we hit the `incr == 0` clause in the last reduction loop,
    // one of the two slices is guaranteed to be at most, two elements.
    assert!(smallest.num_elems() <= 2 || largest.num_elems() <= 2);

    // At this point, we switch gears. smallest and largest we're used to
    // indicate the leftward or rightward median values. Now that we're
    // finishing the final median, we only need all of the fewest elements
    // as well as fewest * 2 elements from most centered around the median of
    // most. Since we're only inserting up to 2 elements, the final median can
    // only be that far away from the current median.
    let (fewest, most) = if smallest.num_elems() < largest.num_elems() {
        (smallest, largest)
    } else {
        (largest, smallest)
    };

    let mut tuples = (fewest.median_idx_min..=fewest.median_idx_max)
        .map(|median_idx| (fewest.list_idx, median_idx))
        .collect::<Vec<_>>();

    assert!(!tuples.is_empty());

    // If the number of elements in most is even, we want it to remain even,
    // this bonus accounts for that.
    let bonus = if (most.median_idx_max - most.median_idx_min + 1) % 2 == 0 {
        1
    } else {
        0
    };
    let most_median_idx = most.median_idx_right();
    let most_min = std::cmp::max(
        most.median_idx_min,
        most_median_idx - std::cmp::min(most_median_idx, tuples.len() + bonus),
    );
    let most_max =
        std::cmp::min(most.median_idx_max, most_median_idx + tuples.len());

    tuples.extend(
        (most_min..=most_max).map(|median_idx| (most.list_idx, median_idx)),
    );

    // Now a simple sort and median calculation.
    tuples.sort_by(|lhs, rhs| lists[lhs.0][lhs.1].cmp(&lists[rhs.0][rhs.1]));

    if tuples.len() % 2 == 0 {
        Some(tuples[(tuples.len() + 1) / 2])
    } else {
        Some(tuples[tuples.len() / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_failure_001() {
        let input = vec![vec![], vec![1i64]];
        assert!(check_values(input));
    }

    #[test]
    fn check_failure_002() {
        let input = vec![vec![0], vec![-1, -1, 0]];
        assert!(check_values(input));
    }

    #[test]
    fn check_failure_003() {
        let input = vec![vec![0, 0], vec![-1, -1, -1, -1, 0, 0, 0]];
        assert!(check_values(input));
    }

    #[test]
    fn check_failure_004() {
        let input =
            vec![vec![1], vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]];
        assert!(check_values(input));
    }

    #[test]
    fn check_failure_005() {
        let input = vec![vec![0], vec![1, 1, -1, -1]];
        assert!(check_values(input));
    }

    #[test]
    fn check_failure_006() {
        let input = vec![vec![0, 1]];
        assert!(check_values(input));
    }

    #[test]
    fn check_failure_007() {
        let input = vec![vec![0], vec![-1, 0]];
        assert!(check_values(input));
    }

    #[test]
    fn check_failure_008() {
        let input = vec![vec![-2, 0], vec![-1]];
        assert!(check_values(input));
    }

    #[test]
    fn check_failure_009() {
        let input = vec![vec![], vec![0], vec![0, 1]];
        assert!(check_values(input));
    }

    #[test]
    fn check_failure_010() {
        let input = vec![vec![0, 0], vec![-1, -1, -1, 1]];
        assert!(check_values(input));
    }

    #[test]
    fn check_failure_011() {
        let input = vec![vec![0], vec![0], vec![1]];
        assert!(check_values(input))
    }

    #[test]
    fn check_failure_012() {
        let input = vec![vec![0], vec![0], vec![0, 1]];
        assert!(check_values(input));
    }

    #[test]
    fn check_failure_013() {
        let input = vec![vec![0], vec![0], vec![-1, 0]];
        assert!(check_values(input));
    }

    #[test]
    fn check_failure_014() {
        let input = vec![vec![], vec![0], vec![0], vec![1, 1, -1, -1]];
        assert!(check_values(input));
    }

    quickcheck::quickcheck! {
        fn qc_multi_median(values: Vec<Vec<i64>>) -> bool {
            check_values(values)
        }
    }

    fn check_values(values: Vec<Vec<i64>>) -> bool {
        // Make sure that each input vector is sorted
        let mut values = values;
        values.iter_mut().for_each(|vals| vals.sort());

        let slices =
            values.iter().map(|list| list.as_ref()).collect::<Vec<_>>();

        let median = find(&slices);
        if slices.is_empty() || slices.iter().all(|list| list.is_empty()) {
            assert!(median.is_none());
            return true;
        }

        assert!(median.is_some());
        let median = median.unwrap();

        let sorted = crate::merge::merge_sort(values.clone());
        assert!(!sorted.is_empty());

        let linfo = ListInfo::new(0, sorted.len());
        let sorted_median = sorted[linfo.median_idx_right()];

        // eprintln!("{:#?}", sorted);
        // eprintln!(
        //     "{} {} = {}: {}",
        //     median.0, median.1, values[median.0][median.1], sorted_median
        // );
        values[median.0][median.1] == sorted_median
    }
}
