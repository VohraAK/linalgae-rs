use std::fmt::{self, Display};
use std::ops::{AddAssign, Fn};
use rand::prelude::*;
use rand_distr::{StandardNormal, Distribution};
use num_traits::Num;

// defining a matrix struct (generic type)
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T>
where T : Num
{
    rows: usize,
    cols: usize,
    data: Vec<T>
}

impl<T> Matrix<T>
where T: Num
{
    // "default" constructor
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, &'static str>
    {
        // check rows and cols
        if rows == 0 { return Err("Row dim should be positive!"); }

        if cols == 0 { return Err("Col dim should be positive!"); }

        // check if number of elements equal to (rows x cols)
        if (rows * cols) != data.len() { return Err("Input vector does not match dimensions!"); }
        
        Ok(Matrix { rows, cols, data })
    }

    // "full" constructor, returns a Matrix filled with <fill_value>
    pub fn full(rows: usize, cols: usize, fill_value: T) -> Result<Self, &'static str>
    where T: Clone
    {
        // check rows and cols
        if rows == 0 { return Err("Row dim should be positive!"); }

        if cols == 0 { return Err("Col dim should be positive!"); }

        Ok(Matrix { rows, cols, data: vec![fill_value; rows * cols] })
    }

    // "zeros" constructor, returns a zeroed out Matrix
    pub fn zeroes(rows: usize, cols: usize) -> Result<Self, &'static str> 
    where T: Num + Clone + Copy
    {
        Self::full(rows, cols, T::zero())
    }

    // "ones" constructor, returns a Matrix filled with ones
    pub fn ones(rows: usize, cols: usize) -> Result<Self, &'static str> 
    where T: Clone + num_traits::One
    {
        Self::full(rows, cols, T::one()) 
    }

    // "identity" constructor, returns an identity Matrix
    pub fn identity(n: usize) -> Result<Self, &'static str>
    where T: Num + Clone + Copy
    {
        if n == 0 { return Err("Dimension should be positive!"); }

        let mut mat = Self::zeroes(n, n)?;
        let data = mat.as_mut_slice();

        for i in 0..n
        {
            data[i*n + i] = T::one();
        }

        Ok(mat)
    }

    // dim getters
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    // data accessors implemented as slicers
    pub fn as_slice(&self) -> &[T] { &self.data }
    pub fn as_mut_slice(&mut self) -> &mut[T] { &mut self.data }

    // transpose function
    pub fn t(&self) -> Matrix<T>
    where T: Default + Clone + Copy,
    {
        let rows = self.rows();
        let cols = self.cols();
        let mut result = vec![T::zero(); rows * cols];
        let slice = self.as_slice();

        for i in 0..rows 
        { 
            for j in 0..cols 
            { 
                result[j*rows +i] = slice[i*cols + j]; 
            } 
        }

        Matrix::new(cols, rows, result).unwrap()

    }

    // apply function -> applies a closure / function to all values of the matrix
    // using Fn syntax since it can both capture variables from outer scopes, and act as a regular function
    pub fn apply(&self, func: impl Fn(T) -> T) -> Matrix<T>
    where T: Copy
    {
        let result = self.data.as_slice().iter().map(|x| func(*x)).collect();

        Matrix::new(self.rows, self.cols, result).unwrap()
    }

    // matmul function
    pub fn matmul(&self, other: &Matrix<T>) -> Matrix<T>
    where T: Num + Copy + AddAssign
    {
        let lhs_cols = self.cols();
        let rhs_rows = other.rows();
        
        // check size
        if lhs_cols != rhs_rows
        {
            panic!("Matrix::matmul: lhs_cols ({}) != rhs_rows ({})", lhs_cols, rhs_rows);
        }

        let lhs_rows = self.rows();
        let rhs_cols = other.cols();

        let lhs_slice = self.as_slice();
        let rhs_slice = other.as_slice();

        let mut result = vec![T::zero(); lhs_rows * rhs_cols];

        // optimised version of matmul, should be much faster
        for i in 0..lhs_rows 
        {
            for k in 0..lhs_cols 
            {
                let lhs_val = lhs_slice[i * lhs_cols + k];
                
                for j in 0..rhs_cols 
                {
                    let rhs_val = rhs_slice[k * rhs_cols + j];
                    let res_idx = i * rhs_cols + j;
                    result[res_idx] += lhs_val * rhs_val;
                }
            }
        }

        Matrix::new(lhs_rows, rhs_cols, result).expect("Matrix::matmul: Result has incorrect dimensions!")
    }

    // Hadamard multiplication
    pub fn component_mul(&self, other: &Matrix<T>) -> Matrix<T>
    where T: Num + Copy
    {
        let rows = self.rows();
        let cols = self.cols();
        let other_rows = other.rows();
        let other_cols = other.cols();

        // check size
        if rows != other_rows || cols != other_cols
        {
            panic!("Matrix::component_mul: Dimension mismatch!")
        }

        let result = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a * b).collect();

        Matrix::new(rows, cols, result).expect("Matrix::component_mul: Error!")
        
    }

    // sum func
    pub fn sum(&self) -> T
    where T: Copy + std::iter::Sum<T>
    {
        self.data.iter().copied().sum()
    }
}

impl<T> Matrix<T>
where T: Copy + Num,
StandardNormal: Distribution<T>
{
    pub fn rand_init(rows: usize, cols: usize) -> Matrix<T>
    {
        // check rows and cols
        if rows == 0 { panic!("Row dim should be positive!"); }

        if cols == 0 { panic!("Col dim should be positive!"); }

        let mut rng = rand::rng();

        let data = vec![T::zero(); rows * cols].iter().map(|_| rng.sample(StandardNormal)).collect();

        Matrix::new(rows, cols, data).expect("Martix::rand_init: Error!")

    }
}

// X----------X
// impl fmt::Display trait for Matrix
impl<T> fmt::Display for Matrix<T>
where T: Num + Display
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result 
    {
        write!(f, "\n[")?;

        for row in 0..self.rows
        {
            write!(f, "[")?;

            for col in 0..self.cols
            {
                let idx = row * self.cols + col;
                write!(f, "{:.2}", self.data[idx])?;

                if col != self.cols - 1
                {
                    write!(f, ", ")?;
                }
            }

            write!(f, "]")?;

            if row != self.rows - 1
            {
                write!(f, ",\n")?;
            }
        }

        write!(f, "]")?;

        Ok(())
    }
}