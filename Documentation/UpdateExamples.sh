#!/bin/bash

# Remove Examples directory if it exists and is not a symlink
if [ -d "Examples" ] && [ ! -L "Examples" ]; then
    rm -rf Examples
fi

# Create a symbolic link to ../Examples if it doesn't exist
if [ ! -L "Examples" ]; then
    ln -s ../Examples Examples
fi

# Clean old Sphinx build
make clean

# Build new HTML
make html

# Move up one directory (to PyOR root)
cd ..

# Make sure docs/ folder exists
mkdir -p docs

# Copy new HTML files into docs/, overwrite old files
cp -r Documentation/_build/html/* docs/

# Create .nojekyll file inside docs/
touch docs/.nojekyll

