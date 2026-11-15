import numpy as np

def chi2_independence(C: list) -> dict:
    """
    Returns a dictionary with chi2 and expected.
    """
    # Convert input list to a NumPy float array
    O = np.asarray(C, dtype=float)
    
    # Compute row totals, column totals, and grand total
    row_totals = np.sum(O, axis=1)
    col_totals = np.sum(O, axis=0)
    grand_total = np.sum(O)
    
    # Build the expected counts table using an outer product
    expected = np.outer(row_totals, col_totals) / grand_total
    
    # Compute the chi-square statistic and cast to a standard Python float
    chi2 = float(np.sum((O - expected) ** 2 / expected))
    
    return {
        "chi2": chi2,
        "expected": expected
    }