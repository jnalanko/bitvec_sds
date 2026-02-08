use std::{io::{Cursor, Read}, sync::Arc};
use bitvec::prelude::*;
use rand_xoshiro::{Xoshiro256PlusPlus, rand_core::{RngCore, SeedableRng}};

use bitvec_sds::{rank_support_v::RankSupportV, traits::{Pat1, Sel1}};
use simple_sds_sbwt::{ops::{Rank, Select}, serialize::Serialize};

fn main() {

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
    let n = 8_000_000_000usize; // Is a multiple of 64
    let mut bv_data = Vec::<u64>::with_capacity(n/64);
    println!("Generating random bitvector of length {n}...");
    for _ in 0..n/64 {
        bv_data.push(rng.next_u64());
    }

    let bv = BitVec::<u64, Lsb0>::from_vec(bv_data);
    let bv = Arc::new(bv);
    println!("{}", bv.len());
    benchmark_rank(bv.clone());
    benchmark_select(bv.clone());
}

fn benchmark_rank(bv: Arc<BitVec<u64, Lsb0>>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(131313);
    let n = bv.len();

    println!("Building RankSupportV");
    let rank_support = RankSupportV::<Pat1>::new(bv.clone());

    println!("Building simple_sds_sbwt rank support");
    let simple_sds = bitvec_to_simple_sds_raw_bitvec((*bv).clone());
    let mut simple_sds = simple_sds_sbwt::bit_vector::BitVector::from(simple_sds);
    simple_sds.enable_rank();

    // Draw 1 million random query positions
    let n_queries = 1_000_000;
    println!("Generating {n_queries} random query positions...");
    let mut query_positions: Vec<usize> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let pos = rng.next_u64() as usize % n;
        query_positions.push(pos);
    }

    println!("Running rank queries on RankSupportV");

    // Start the timer
    let start = std::time::Instant::now();

    // Run the queries
    let mut sum_of_answers = 0_usize;
    for &pos in &query_positions {
        sum_of_answers += rank_support.rank(pos);
        //sum_of_answers += rank_support.rank_AI_optimized(pos);
        //unsafe { sum_of_answers += rank_support.rank_unchecked(pos); }
    }
    println!("Sum of all rank answers: {}", sum_of_answers);

    // Print the elapsed time per query in nanoseconds
    let elapsed = start.elapsed();
    let avg_time_per_query = elapsed.as_secs_f64() / n_queries as f64;
    println!("Average time per rank query: {:.2} ns", avg_time_per_query * 1e9);

    println!("Running rank queries on simple_sds bit vector");

    // Start the timer
    let start = std::time::Instant::now();

    // Run the queries
    let mut sum_of_answers = 0_usize;
    for &pos in &query_positions {
        sum_of_answers += simple_sds.rank(pos);
    }
    println!("Sum of all rank answers: {}", sum_of_answers);

    // Print the elapsed time per query in nanoseconds
    let elapsed = start.elapsed();
    let avg_time_per_query = elapsed.as_secs_f64() / n_queries as f64;
    println!("Average time per rank query: {:.2} ns", avg_time_per_query * 1e9);
}

fn benchmark_select(bv: Arc<BitVec<u64, Lsb0>>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(131313);
    let n_ones = bv.count_ones();


    // Benchmark select queries
    let n_queries = 1_000_000;
    println!("Building SelectSupportMcl");
    let select_support = bitvec_sds::select_support_mcl::SelectSupportMcl::<Sel1>::new(bv.clone());

    println!("Building simple_sds_sbwt select support");
    let simple_sds = bitvec_to_simple_sds_raw_bitvec((*bv).clone());
    let mut simple_sds = simple_sds_sbwt::bit_vector::BitVector::from(simple_sds);
    simple_sds.enable_select();

    println!("Generating {n_queries} random queries");
    let max_select_query = n_ones;
    let mut select_query_positions: Vec<usize> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let pos = rng.next_u64() as usize % max_select_query;
        select_query_positions.push(pos);
    }

    println!("Running select queries on SelectSupportMcl");

    // Start the timer
    let start = std::time::Instant::now();

    // Run the queries
    let mut sum_of_select_answers = 0_usize;
    for (query_idx, query) in select_query_positions.iter().enumerate() {
        let ans = select_support.select(*query);
        sum_of_select_answers += ans;
        if query_idx < 10 {
            eprintln!("{} {}", query_idx, ans);
        }
    }
    println!("Sum of all select answers: {}", sum_of_select_answers);

    // Print the elapsed time per query in nanoseconds
    let elapsed = start.elapsed();
    let avg_time_per_select_query = elapsed.as_secs_f64() / n_queries as f64;
    println!("Average time per select query: {:.2} ns", avg_time_per_select_query * 1e9);

    println!("Running select queries on simple_sds bit vector");

    // Start the timer
    let start = std::time::Instant::now();

    // Run the queries
    let mut sum_of_select_answers = 0_usize;
    for (query_idx, query) in select_query_positions.iter().enumerate() {
        let ans = simple_sds.select(*query - 1).unwrap(); // simple_sds does 1-based indexing in select
        sum_of_select_answers += ans;
        if query_idx < 10 {
            eprintln!("{} {}", query_idx, ans);
        }
    }
    println!("Sum of all select answers: {}", sum_of_select_answers);

    // Print the elapsed time per query in nanoseconds
    let elapsed = start.elapsed();
    let avg_time_per_query = elapsed.as_secs_f64() / n_queries as f64;
    println!("Average time per select query: {:.2} ns", avg_time_per_query * 1e9);

}

fn bitvec_to_simple_sds_raw_bitvec(mut bv: bitvec::vec::BitVec::<u64, Lsb0>) -> simple_sds_sbwt::raw_vector::RawVector {
    // TODO: We really hope that usize equals u64 here, otherwise this this is probably broken.
    // Let's use the deserialization function in simple_sds_sbwt for a raw bitvector.
    // It requires the following header:
    let mut header = [0u64, 0u64]; // bits, words
    header[0] = bv.len() as u64; // Assumes little-endian byte order
    header[1] = bv.len().div_ceil(64) as u64;

    let header_bytes = bytemuck::cast_slice(&header);

    // Make sure the leftover bits in the last word are zeros. Simple-sds
    // depends on this, but the bitvec crate does not guarantee this!
    // The undefined padding bytes have broken my code before, so this is
    // crucial.
    let original_len = bv.len();
    bv.resize(original_len.next_multiple_of(64), false);
    bv.resize(original_len, false);

    let raw_data = bytemuck::cast_slice(bv.as_raw_slice());
    let mut data_with_header = Cursor::new(header_bytes).chain(Cursor::new(raw_data));

    simple_sds_sbwt::raw_vector::RawVector::load(&mut data_with_header).unwrap()
}