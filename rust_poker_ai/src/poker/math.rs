pub fn count_combinations(n: usize, r: usize) -> usize {
    if r > n {
        0
    } else {
        (1..=r).fold(1, |acc, val| acc * (n - val + 1) / val)
    }
}

// pub fn count_permutations(n: usize, r: usize) -> usize {
//     (n - r + 1..=n).product()
// }
