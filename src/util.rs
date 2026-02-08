
static LOW_MASK: [u64; 65] = [
    0x0000000000000000, 0x0000000000000001, 0x0000000000000003, 0x0000000000000007,
    0x000000000000000F, 0x000000000000001F, 0x000000000000003F, 0x000000000000007F,
    0x00000000000000FF, 0x00000000000001FF, 0x00000000000003FF, 0x00000000000007FF,
    0x0000000000000FFF, 0x0000000000001FFF, 0x0000000000003FFF, 0x0000000000007FFF,
    0x000000000000FFFF, 0x000000000001FFFF, 0x000000000003FFFF, 0x000000000007FFFF,
    0x00000000000FFFFF, 0x00000000001FFFFF, 0x00000000003FFFFF, 0x00000000007FFFFF,
    0x0000000000FFFFFF, 0x0000000001FFFFFF, 0x0000000003FFFFFF, 0x0000000007FFFFFF,
    0x000000000FFFFFFF, 0x000000001FFFFFFF, 0x000000003FFFFFFF, 0x000000007FFFFFFF,
    0x00000000FFFFFFFF, 0x00000001FFFFFFFF, 0x00000003FFFFFFFF, 0x00000007FFFFFFFF,
    0x0000000FFFFFFFFF, 0x0000001FFFFFFFFF, 0x0000003FFFFFFFFF, 0x0000007FFFFFFFFF,
    0x000000FFFFFFFFFF, 0x000001FFFFFFFFFF, 0x000003FFFFFFFFFF, 0x000007FFFFFFFFFF,
    0x00000FFFFFFFFFFF, 0x00001FFFFFFFFFFF, 0x00003FFFFFFFFFFF, 0x00007FFFFFFFFFFF,
    0x0000FFFFFFFFFFFF, 0x0001FFFFFFFFFFFF, 0x0003FFFFFFFFFFFF, 0x0007FFFFFFFFFFFF,
    0x000FFFFFFFFFFFFF, 0x001FFFFFFFFFFFFF, 0x003FFFFFFFFFFFFF, 0x007FFFFFFFFFFFFF,
    0x00FFFFFFFFFFFFFF, 0x01FFFFFFFFFFFFFF, 0x03FFFFFFFFFFFFFF, 0x07FFFFFFFFFFFFFF,
    0x0FFFFFFFFFFFFFFF, 0x1FFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
];

#[inline]
pub fn low_mask(k: u32) -> u64 {
    LOW_MASK[k as usize]
}

#[inline]
pub fn word_at(words: &[u64], len_bits: usize, widx: usize) -> u64 {
    if widx >= words.len() {
        return 0;
    }
    let mut w = words[widx];
    if widx + 1 == words.len() {
        let rem = len_bits & 63;
        if rem != 0 {
            w &= low_mask(rem as u32); // Lsb0: keep low `rem` bits
        }
    }
    w
}


/// Return the bit position (0..63) of the i-th 1 in `word` (i is 1-based).
#[inline]
pub fn select_in_word(mut word: u64, i: u32) -> u32 {
    debug_assert!(i >= 1);
    // clear i-1 low 1-bits
    for _ in 1..i {
        word &= word - 1;
    }
    word.trailing_zeros()
}

#[inline]
pub fn map10(w: u64, carry: u64) -> u64 {
    // Same mapping used by SDSL: map10 = (w ^ ((w<<1)|carry)) & (~w)
    (w ^ ((w << 1) | (carry & 1))) & (!w)
}

#[inline]
pub fn map01(w: u64, carry: u64) -> u64 {
    // map01 = (w ^ ((w<<1)|carry)) & w
    (w ^ ((w << 1) | (carry & 1))) & w
}

#[inline]
pub fn lo_unset(rem: usize) -> u64 {
    // SDSL's bits::lo_unset[rem] for rem in 0..64:
    // mask that is 1 for bits >= rem (i.e., high bits), 0 for low rem bits.
    // We'll use it to subtract mapped hits in the unused high tail of the last word.
    debug_assert!(rem < 64);
    !low_mask(rem as u32)
}

#[inline]
pub fn mask_last_word(words: &[u64], len_bits: usize, widx: usize) -> u64 {
    let mut w = words[widx];
    if widx + 1 == words.len() {
        let rem = len_bits & 63;
        if rem != 0 {
            w &= low_mask(rem as u32);
        }
    }
    w
}
