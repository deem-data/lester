package berlin.deem.lester.pipeline

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class Graph {

  private val in: mutable.HashMap[String, ArrayBuffer[String]] = mutable.HashMap.empty
  private val out: mutable.HashMap[String, ArrayBuffer[String]] = mutable.HashMap.empty
  private val weights: mutable.HashMap[(String, String), Int] = mutable.HashMap.empty

  def addNode(node: String): Unit = {
    if (!in.contains(node)) {
      in(node) = ArrayBuffer.empty
    }
    if (!out.contains(node)) {
      out(node) = ArrayBuffer.empty
    }
  }

  def nodes(): Iterator[String] = {
    in.keysIterator
  }

  def addEdge(source: String, target: String): Unit = {
    out(source).append(target)
    in(target).append(source)
  }

  def addWeight(source: String, target: String, weight: Int): Unit = {
    weights((source, target)) = weight
  }

  def weight(source: String, target: String): Option[Int] = {
    weights.get(source, target)
  }

  def inDegree(node: String): Int = {
    in(node).length
  }

  def outDegree(node: String): Int = {
    out(node).length
  }

  def successors(node: String): Iterator[String] = {
    out(node).iterator
  }

  def predecessors(node: String): Iterator[String] = {
    in(node).iterator
  }

}
