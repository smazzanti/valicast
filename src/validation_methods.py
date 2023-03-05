import numpy as np

# main function

def get_indices(method:str, time_series_length:int, **kwargs):
  """
  Get train and test indices for the given method.
  
  Args:
    method: Name of the method.
      Valid choices are: ("holdout", "inv_holdout", "rep_holdout", "cv", "cv_bl", "cv_mod", "cv_hvbl", 
      "preq_bls", "preq_sld_bls", "preq_bls_gap", "preq_slide", "preq_grow")
    time_series_length: Length of the time series.
    kwargs: Additional arguments, depending on the chosen method.
      In detail, for each method the required arguments are:
      - "holdout" -> "train_size"
      - "inv_holdout" -> "train_size"
      - "rep_holdout" -> "n_reps", "train_size", "test_size" 
      - "cv" -> "n_folds"
      - "cv_bl" -> "n_folds"
      - "cv_mod" -> "n_folds", "gap_before", "gap_after" 
      - "cv_hvbl" -> "n_folds", "gap_before", "gap_after"
      - "preq_bls" -> "n_folds"
      - "preq_sld_bls" -> n_folds" 
      - "preq_bls_gap" -> "n_folds" 
      - "preq_slide" -> "train_size", "n_reps"
      - "preq_grow" -> "train_size", "n_reps"
    
  Yields:
    Train index and test index.
  """
  function_name = f"get_indices_{method}"
  return eval(function_name)(time_series_length=time_series_length, **kwargs)  


# method 1. holdout

def get_indices_holdout(time_series_length:int, train_size:float):
  """
  Get train and test index for method: "Holdout".
  
  Args:
    time_series_length: Length of the time series.
    train_size: Size of training set, as a fraction of the time series.
    
  Returns:
    Train index and test index.
  """
  
  cut_point = np.round(time_series_length * train_size)
  train_id = np.arange(start=0, stop=cut_point, step=1, dtype=int)
  test_id = np.arange(start=cut_point, stop=time_series_length, step=1, dtype=int)

  yield train_id, test_id
    
    
# method 2. inv-holdout

def get_indices_inv_holdout(time_series_length:int, train_size:float):
  """
  Get train and test index for method: "Inverse Holdout".
  
  Args:
    time_series_length: Length of the time series.
    train_size: Size of training set, as a fraction of the time series.
    
  Returns:
    Train index and test index.
  """
  
  cut_point = np.round(time_series_length * (1 - train_size))
  test_id = np.arange(start=0, stop=cut_point, step=1, dtype=int)
  train_id = np.arange(start=cut_point, stop=time_series_length, step=1, dtype=int)

  yield train_id, test_id


# method 3. rep-holdout

def get_indices_rep_holdout(time_series_length:int, n_reps:int, train_size:float, test_size:float):
  """
  Get train and test index for method: "Repeated Holdout".
  
  Args:
    time_series_length: Length of the time series.
    n_reps: Number of repetitions.
    train_size: Size of training set, as a fraction of the time series.
    test_size: Size of test set, as a fraction of the time series.
    
  Yields:
    Train index and test index.
  """

  train_length = np.round(time_series_length * train_size, 0)
  test_length = np.round(time_series_length * test_size, 0)
  valid_cut_points = np.arange(start=train_length, stop=time_series_length-test_length+1, step=1, dtype=int)

  for rep in range(n_reps):
    cut_point = np.random.choice(valid_cut_points)
    train_id = np.arange(start=cut_point-train_length, stop=cut_point, step=1, dtype=int)
    test_id = np.arange(start=cut_point, stop=cut_point+test_length, step=1, dtype=int)
    
    yield train_id, test_id
    
    
# method 4. cv (plain)

def get_indices_cv(time_series_length:int, n_folds:int):
  """
  Get train and test index for method: "Cross-Validation".
  
  Args:
    time_series_length: Length of the time series.
    n_folds: Number of folds.
    
  Yields:
    Train index and test index.
  """

  test_fold = np.random.permutation(adjacent_folds(time_series_length=time_series_length, n_folds=n_folds))

  for fold in range(n_folds):
    train_id = np.where(test_fold != fold)[0]
    test_id = np.where(test_fold == fold)[0]
    yield train_id, test_id
    

# method 5. cv-blocked

def get_indices_cv_bl(time_series_length:int, n_folds:int):
  """
  Get train and test index for method: "Blocked Cross-Validation".
  
  Args:
    time_series_length: Length of the time series.
    n_folds: Number of folds.
    
  Yields:
    Train index and test index.
  """

  test_fold = adjacent_folds(time_series_length=time_series_length, n_folds=n_folds)

  for fold in range(n_folds):
    train_id = np.where(test_fold != fold)[0]
    test_id = np.where(test_fold == fold)[0]
    yield train_id, test_id
    

# method 6. cv-mod

def get_indices_cv_mod(time_series_length:int, n_folds:int, gap_before:int, gap_after:int):
  """
  Get train and test index for method: "Modified Cross-Validation".
  
  Args:
    time_series_length: Length of the time series.
    n_folds: Number of folds.
    gap_before: Number of time points to be discarded, before the cut point.
    gap_after: Number of time points to be discarded, after the cut point.
    
  Yields:
    Train index and test index.
  """

  test_fold = np.random.permutation(adjacent_folds(time_series_length=time_series_length, n_folds=n_folds))

  for fold in range(n_folds):
    test_id = np.where(test_fold == fold)[0]
    train_id = np.where([fold not in test_fold[max(i-gap_before,0):min(i+gap_after+1,time_series_length)] for i in range(time_series_length)])[0]
    yield train_id, test_id
    

# method 7. cv-hvBl

def get_indices_cv_hvbl(time_series_length:int, n_folds:int, gap_before:int, gap_after:int):
  """
  Get train and test index for method: "Hv-Blocked Cross-Validation".
  
  Args:
    time_series_length: Length of the time series.
    n_folds: Number of folds.
    gap_before: Number of time points to be discarded, before the cut point.
    gap_after: Number of time points to be discarded, after the cut point.
    
  Yields:
    Train index and test index.
  """

  test_fold = adjacent_folds(time_series_length=time_series_length, n_folds=n_folds)

  for fold in range(n_folds):
    test_id = np.where(test_fold == fold)[0]
    train_id = np.where([fold not in test_fold[max(i-gap_before,0):min(i+gap_after+1,time_series_length)] for i in range(time_series_length)])[0]
    yield train_id, test_id
    

# method 8. preq-bls

def get_indices_preq_bls(time_series_length:int, n_folds:int):
  """
  Get train and test index for method: "Prequential Blocks".
  
  Args:
    time_series_length: Length of the time series.
    n_folds: Number of folds.
    
  Yields:
    Train index and test index.
  """

  test_fold = adjacent_folds(time_series_length=time_series_length, n_folds=n_folds)

  for fold in range(1, n_folds):
    test_id = np.where(test_fold == fold)[0]
    train_id = np.where(test_fold < fold)[0]
    yield train_id, test_id
    

# method 9. Preq-Sld-Bls

def get_indices_preq_sld_bls(time_series_length:int, n_folds:int):
  """
  Get train and test index for method: "Prequential Sliding Blocks".
  
  Args:
    time_series_length: Length of the time series.
    n_folds: Number of folds.
    
  Yields:
    Train index and test index.
  """

  test_fold = adjacent_folds(time_series_length=time_series_length, n_folds=n_folds)

  for fold in range(1, n_folds):
    test_id = np.where(test_fold == fold)[0]
    train_id = np.where(test_fold == fold - 1)[0]
    yield train_id, test_id
    

# method 10. Preq-Bls-Gap

def get_indices_preq_bls_gap(time_series_length:int, n_folds:int):
  """
  Get train and test index for method: "Prequential Sliding Blocks with Gap".
  
  Args:
    time_series_length: Length of the time series.
    n_folds: Number of folds.
    
  Yields:
    Train index and test index.
  """

  test_fold = adjacent_folds(time_series_length=time_series_length, n_folds=n_folds)

  for fold in range(2, n_folds):
    test_id = np.where(test_fold == fold)[0]
    train_id = np.where(test_fold < fold - 1)[0]
    yield train_id, test_id
    

# method 11. Preq-Slide

def get_indices_preq_slide(time_series_length:int, train_size:float, n_reps:int):
  """
  Get train and test index for method: "Prequential Sliding".
  
  Args:
    time_series_length: Length of the time series.
    train_size: Size of training set, as a fraction of the time series.
    n_reps: Maximum number of repetitions.
    
  Yields:
    Train index and test index.
  """

  train_length = int(np.round(time_series_length * train_size, 0))
  n_reps = min(n_reps, time_series_length-train_length)
  cut_points = np.linspace(start=train_length, stop=time_series_length, num=n_reps+1, dtype=int)[:n_reps]

  for cut_point in cut_points:
    train_id = np.arange(start=cut_point-train_length, stop=cut_point, step=1, dtype=int)
    test_id = np.arange(start=cut_point, stop=time_series_length, step=1, dtype=int)
    yield train_id, test_id
    
    
# method 12. Preq-Grow

def get_indices_preq_grow(time_series_length:int, train_size:float, n_reps:int):
  """
  Get train and test index for method: "Prequential Growing".
  
  Args:
    time_series_length: Length of the time series.
    train_size: Size of training set, as a fraction of the time series.
    n_reps: Maximum number of repetitions.
    
  Yields:
    Train index and test index.
  """

  train_length = int(np.round(time_series_length * train_size, 0))
  n_reps = min(n_reps, time_series_length-train_length)
  cut_points = np.linspace(start=train_length, stop=time_series_length, num=n_reps+1, dtype=int)[:n_reps]

  for cut_point in cut_points:
    train_id = np.arange(start=0, stop=cut_point, step=1, dtype=int)
    test_id = np.arange(start=cut_point, stop=time_series_length, step=1, dtype=int)
    yield train_id, test_id
    

# util

def adjacent_folds(time_series_length: int, n_folds: int):
  """
  Return adjacent folds.
  
  Args:
    time_series_length: Length of the time series.
    n_folds: Number of folds.
    
  Returns:
    Array containing the identifier of each fold.
  
  Example: 
    adjacent_folds(time_series_length=10, n_folds=4) -> array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
  """

  div, mod = divmod(time_series_length, n_folds)
  arr = np.array(sum([[fold] * (div + int(fold < mod)) for fold in range(n_folds)], []))
  return arr