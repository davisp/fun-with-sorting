//! Sorting FTW!
//!
//! This module contains a few different implementations of merge sort for
//! inputs that consist of pre-sorted values.

pub fn merge_sort(mut input: Vec<Vec<i64>>) -> Vec<i64> {
    if input.is_empty() {
        return Default::default();
    }

    if input.len() == 1 {
        return input.pop().unwrap();
    }

    if input.len() == 2 {
        let mut ret = Vec::with_capacity(input[0].len() + input[1].len());
        let mut lhs = 0;
        let mut rhs = 0;

        while lhs < input[0].len() && rhs < input[1].len() {
            if lhs > 0 {
                assert!(input[0][lhs] >= input[0][lhs - 1]);
            }
            if rhs > 0 {
                assert!(input[1][rhs] >= input[1][rhs - 1]);
            }
            if input[0][lhs] < input[1][rhs] {
                ret.push(input[0][lhs]);
                lhs += 1;
            } else {
                ret.push(input[1][rhs]);
                rhs += 1;
            }
        }

        while lhs < input[0].len() {
            if lhs > 0 {
                assert!(input[0][lhs] >= input[0][lhs - 1]);
            }
            ret.push(input[0][lhs]);
            lhs += 1;
        }

        while rhs < input[1].len() {
            if rhs > 0 {
                assert!(input[1][rhs] >= input[1][rhs - 1]);
            }
            ret.push(input[1][rhs]);
            rhs += 1;
        }

        return ret;
    }

    let middle = input.len() / 2;
    let right = input.split_off(middle);

    assert!(!input.is_empty());
    assert!(!right.is_empty());

    let lhs = merge_sort(input);
    let rhs = merge_sort(right);

    merge_sort(vec![lhs, rhs])
}

#[cfg(test)]
mod tests {
    use super::*;

    quickcheck::quickcheck! {
        fn qc_merge_sort(values: Vec<Vec<i64>>) -> bool {
            // Make sure that each input vector is sorted
            let mut values = values;
            values.iter_mut().for_each(|vals| vals.sort());

            let sorted = merge_sort(values);
            sorted.windows(2).all(|pair| {
                pair[0] <= pair[1]
            })
        }
    }
}
