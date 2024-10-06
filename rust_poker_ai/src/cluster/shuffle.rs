use rand::Rng;
use rand::rngs::ThreadRng;


pub fn get_random_index(rng: &mut ThreadRng, length: usize) -> usize {
    rng.gen_range(0..length)
}

pub fn get_random_index_pair(rng: &mut ThreadRng, length: usize) -> (usize, usize) {
    let index1 = rng.gen_range(0..length) as usize;
    let mut index2 = rng.gen_range(0..(length - 1)) as usize;
    if index1 == index2 {
        index2 += 1;
    }
    
    (index1, index2)
}

pub fn get_random_index_pair_with_except(rng: &mut ThreadRng, length: usize, except: usize) -> (usize, usize) {
    let mut index1 = rng.gen_range(0..(length - 1)) as usize;
    if index1 == except {
        index1 += 1;
    }
    let mut index2 = rng.gen_range(0..(length - 2)) as usize;
    if index2 == except || index2 == index1 {
        index2 += 1;
    }
    if index2 == except || index2 == index1 {
        index2 += 1;
    }
    
    (index1, index2)
}
