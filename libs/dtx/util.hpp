#pragma once

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#include "dtx.hpp"
using namespace std;

const int TEST = 3;
const vector<int> TILE_SHAPE = {32,32};

enum DataLayout {
    RowMajor = 0,
    ColMajor = 1,
};

// ========================================================
//                      DTX HELPERS
// ========================================================

bool compare_two_vectors_of_ints(vector<int> a, vector<int> b);
bool compare_two_tensors(DTXTensor * a, DTXTensor * b);
bool compare_two_tensor_pairs(TensorPair * a, TensorPair * b);
bool compare_two_groups(TensorPairGroup * a, TensorPairGroup * b);

// ========================================================
//                      TENSOR OVERLAP
// ========================================================
bool has_overlap(DTXTensor * overlap);
vector<int> calculate_line_segment_overlap_in_1d(int l1_str, int l1_end, int l2_str, int l2_end);
DTXTensor * calculate_tensor_overlap_in_nd(DTXTensor * t1, DTXTensor * t2);

int X(int rank);
int Y(int rank);
int Z(int rank);
int W(int rank);
