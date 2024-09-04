fn get_int_rank(raw_rank: u8) -> i32 {
    return (raw_rank as i32) - 2;
}

fn get_int_suit(suit: char) -> i32 {
    match suit {
        's' => 1,
        'h' => 2,
        'd' => 4,
        'c' => 8,
        _ => -1,
    }
}

fn get_prime_rank(rank_int: i32) -> i32 {
    match rank_int {
        0 => 2,
        1 => 3,
        2 => 5,
        3 => 7,
        4 => 11,
        5 => 13,
        6 => 17,
        7 => 19,
        8 => 23,
        9 => 29,
        10 => 31,
        11 => 37,
        12 => 41,
        _ => -1,
    }
}

pub fn new(raw_rank: u8, char_suit: char) -> i32 {
    let int_rank = get_int_rank(raw_rank);
    let int_suit = get_int_suit(char_suit);
    let prime_rank = get_prime_rank(int_rank);

    let bit_rank = 1 << (int_rank as i32) << 16;
    let suit = int_suit << 12;
    let rank = (int_rank as i32) << 8;

    return bit_rank | suit | rank | prime_rank;
}
