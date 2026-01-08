use std::fmt;

// defining a matrix struct (f64)
#[derive(Debug)]
pub struct Matrix
{
    rows: usize,
    cols: usize,
    data: Vec<f64>
}

impl Matrix
{
    // "default" constructor
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, &'static str>
    {
        // check rows and cols
        if rows <= 0
        {
            return Err("Row dim should be positive!");
        }

        if cols <= 0
        {
            return Err("Col dim should be positive!");
        }

        // check if number of elements equal to (rows x cols)
        if (rows * cols) != data.len()
        {
            return Err("Input vector does not match dimensions!");
        }
        
        Ok(Matrix { rows, cols, data })
    }

    // "zeros" constructor, returns a zeroed out Matrix
    pub fn zeroes(rows: usize, cols: usize) -> Result<Self, &'static str>
    {
        // check rows and cols
        if rows <= 0
        {
            return Err("Row dim should be positive!");
        }

        if cols <= 0
        {
            return Err("Col dim should be positive!");
        }

        Ok(Matrix { rows, cols, data: vec![0 as f64; rows * cols] })
    }

    // dim getters
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    // data accessors implemented as slicers
    pub fn as_slice(&self) -> &[f64] { &self.data }
    pub fn as_mut_slice(&mut self) -> &mut[f64] { &mut self.data }

}

// X----------X
// impl fmt::Display trait for Matrix
impl fmt::Display for Matrix
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