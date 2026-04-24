use crate::Error;

/// Source of data for a BIOD run, either transformed from an original dataset
/// or generated from scratch
pub enum DataSource<'src, T> {
    /// Original dataset with a transformation function to produce test datasets
    ///
    /// The length of the transformed datasets MUST be the same as the original dataset
    Transformed {
        original: &'src [T],
        transform: Box<dyn Fn(&[T], u64) -> Vec<T> + 'src>,
    },

    /// Datasets generated from scratch using a provided generator function
    ///
    /// The length of the generated datasets MUST be consistent across calls
    Generated {
        length: usize,
        generator: Box<dyn FnMut(usize, u64) -> Vec<T> + 'src>,
    },

    /// An iterable data source that produces datasets from a provided iterator
    /// The length of the datasets produced MUST be consistent across calls
    ///
    /// If the iterator is exhausted, the test will panic
    Iterable {
        length: usize,
        iter: Box<dyn Iterator<Item = Vec<T>> + 'src>,
    },
}
impl<'src, T> DataSource<'src, T> {
    /// Create a transformed data source from an original dataset and a transformation function
    pub fn transformed<F>(original: &'src [T], transform: F) -> Self
    where
        F: Fn(&[T], u64) -> Vec<T> + 'src,
    {
        Self::Transformed {
            original,
            transform: Box::new(transform),
        }
    }

    /// Create a generated data source from a generator function and the length of datasets to produce
    pub fn generated<F>(length: usize, generator: F) -> Self
    where
        F: FnMut(usize, u64) -> Vec<T> + 'src,
    {
        Self::Generated {
            length,
            generator: Box::new(generator),
        }
    }

    /// Create an iterable data source from a provided iterator
    pub fn iterable<I>(length: usize, iter: I) -> Self
    where
        I: Iterator<Item = Vec<T>> + 'src,
    {
        Self::Iterable {
            length,
            iter: Box::new(iter),
        }
    }

    /// Get the next dataset to test, using the provided seed for any necessary transformations
    /// or generation.
    ///
    /// Can return an error if the data source is exhausted
    pub fn get_data(&mut self, seed: u64) -> Result<Vec<T>, Error> {
        match self {
            DataSource::Transformed {
                original,
                transform,
            } => Ok(transform(original, seed)),
            DataSource::Generated {
                length, generator, ..
            } => Ok(generator(*length, seed)),
            DataSource::Iterable { iter, .. } => iter.next().ok_or(Error::DataSourceExhausted),
        }
    }

    /// Get the length of the datasets produced by this data source
    pub fn len(&self) -> usize {
        match self {
            DataSource::Transformed { original, .. } => original.len(),
            DataSource::Generated { length, .. } => *length,
            DataSource::Iterable { length, .. } => *length,
        }
    }
}

/* From a countable iterator */
impl<'src, T, I> From<I> for DataSource<'src, T>
where
    I: ExactSizeIterator<Item = Vec<T>> + 'src,
{
    fn from(value: I) -> Self {
        Self::iterable(value.len(), value)
    }
}
