use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

const RANDOM_SEED: u64 = 314159;

pub fn new() -> impl RngExt {
    Xoshiro256PlusPlus::seed_from_u64(RANDOM_SEED)
}
