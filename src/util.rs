
#[inline]
pub fn low_mask(k: u32) -> u64 {
    match k {
        0 => 0,
        64 => !0u64,
        _ => (1u64 << k) - 1,
    }
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
