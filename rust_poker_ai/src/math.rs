pub fn factorial(n: usize) -> usize {
    (1..=n).product()
}

pub fn count_combinations(n: usize, r: usize) -> usize {
    factorial(n) / (factorial(r) * factorial(n - r))
}

/// n! / (n - r)!
pub fn count_permutations(n: usize, r: usize) -> usize {
    factorial(n) / factorial(n - r)
}