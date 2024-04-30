package org.apache.spark.ml.linalg

import breeze.linalg.{Vector => BreezeVector}

object BreezeConverter {
  def asBreeze(vector: SparseVector): BreezeVector[Double] = {
    vector.asBreeze
  }
}
