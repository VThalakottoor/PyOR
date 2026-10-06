"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This file contains functions to load crystal orientation data.

    It includes utilities to load data from SIMPSON-formatted CSV files and
    extracted crystallite arrays from SIMPSON's crystallite library:
    https://github.com/vosegaard/simpson/blob/master/crystdat.c

    Each function returns Euler angles alpha, beta, and the associated crystallite weights.
    Gamma is assumed to be 0 and excluded unless explicitly required.
"""


import numpy as np
import pandas as pd
import re
import os


def Load_Crystallite_CSV(filepath, delimiter=",", skiprows=1):
    """
    Loads crystallite orientation data from a SIMPSON-style CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    delimiter : str, optional
        Delimiter used in the file (default is comma).
    skiprows : int, optional
        Number of rows to skip (default is 1 for header).

    Returns
    -------
    alpha : np.ndarray
        Alpha Euler angles (degrees).
    beta : np.ndarray
        Beta Euler angles (degrees).
    gamma : np.ndarray
        Gamma Euler angles (all zeros, degrees).
    weight : np.ndarray
        Solid angle weights (sum should be ~1).
    """
    data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skiprows)
    alpha = data[:, 0]
    beta = data[:, 1]
    gamma = np.zeros_like(alpha)
    weight = data[:, 2]
    return alpha, beta, gamma, weight


def Extract_CrystArray(file_path, array_name):
    """
    Extracts crystallite alpha, beta, and weight values from a C file containing SIMPSON crystallite arrays,
    and saves them to a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the .c file (crystdat.c).
        (Donwload from crystdat.c file from https://github.com/vosegaard/simpson/blob/master/crystdat.c)
    array_name : str
        Name of the crystallite array to extract (e.g., 'rep2000_cryst').
    """
    with open(file_path, "r") as file:
        content = file.read()

    pattern = rf'CRYSTALLITE {array_name}\[\] = \{{(.*?)\}};'  # capture content between braces
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        raise ValueError(f"Array '{array_name}' not found in file '{file_path}'.")

    data_block = match.group(1)
    data = re.findall(r'\{([\d\.\-eE]+),\s*([\d\.\-eE]+),\s*([\d\.\-eE]+)\}', data_block)
    data = np.array(data, dtype=float)

    alpha = data[:, 0]
    beta = data[:, 1]
    weight = data[:, 2]

    # Save to CSV in the same directory as crystdat.c
    output_dir = os.path.dirname(file_path)
    csv_filename = os.path.join(output_dir, f"{array_name}.csv")
    df = pd.DataFrame({"alpha": alpha, "beta": beta, "weight": weight})
    df.to_csv(csv_filename, index=False)
    print(f"Saved {array_name} to {csv_filename}")


def Load_rep2000(file_path):
    """
    Load rep2000_cryst from crystdat.c and save to rep2000_cryst.csv
    """
    Extract_CrystArray(file_path, "rep2000_cryst")


def Load_zcw4180(file_path):
    """
    Load zcw4180_cryst from crystdat.c and save to zcw4180_cryst.csv
    """
    Extract_CrystArray(file_path, "zcw4180_cryst")


def Load_zcw28656(file_path):
    """
    Load zcw28656_cryst from crystdat.c and save to zcw28656_cryst.csv
    """
    Extract_CrystArray(file_path, "zcw28656_cryst")