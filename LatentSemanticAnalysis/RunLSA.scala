// Import required libraries
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._
import scala.io.Source._
import scala.collection.mutable._
import scala.collection.JavaConversions._
import scala.collection.immutable.ListMap
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import java.util.Properties
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.{DenseMatrix => BDenseMatrix, SparseVector => BSparseVector}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, SingularValueDecomposition, Vectors, Vector => MLLibVector}
import org.apache.spark.{SparkConf, SparkContext}
import java.io.{ObjectOutputStream, FileOutputStream, ObjectInputStream, FileInputStream, File}
import java.io.{BufferedWriter, FileWriter}
import scala.reflect.io.Directory
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._


object RunLSA {

  def main(args: Array[String]){
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    if (args.length < 2) {
      System.err.println("Correct arguments: <input-wiki-movie-plots> <input-stopwords>")
      System.exit(1)
    }

    val sparkConf = new SparkConf().setAppName("RunLSA")
    val sc = new SparkContext(sparkConf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    import sqlContext.implicits._

    val startTime = System.nanoTime
    val file = new File("Problem1.txt")
    file.createNewFile()
    val bw = new BufferedWriter(new FileWriter(file))

// Read the title, genre and plot fields of the articles into an initial DataFrame in Spark
    val dataTemp = sqlContext.read.option("delimiter", ",").
      option("header", "true").
      option("inferschema", "true").
      option("multiLine",true).
      option("escape", "\"").
      csv(args(0)).select("title", "genre", "plot")

    // Because some titles have different plots, I will make each title unique so Qn 1E, where we are to get Top5 most
    // frequent genres can be easily solved.
    val data = dataTemp.withColumn("id",monotonicallyIncreasingId).withColumn("title", concat(col("title"), lit(" :"), col("id"))).drop("id")
    
    val numTerms = 5000
    val k = 25
    val numDocs = data.count()
    val bNumDocs = sc.broadcast(numDocs)

// Next, add an additional column called features to your DataFrame, which contains a list of lemmatized text tokens extracted from each of the plot fields using the NLP-based plainTextToLemmas
// function of the given shell script.
    def isOnlyLetters(str: String): Boolean = {
      str.forall(c => Character.isLetter(c))
    }

    val bStopWords = sc.broadcast(
      fromFile(args(1)).getLines().toSet)

    def createNLPPipeline(): StanfordCoreNLP = {
      val props = new Properties()
      props.put("annotators", "tokenize, ssplit, pos, lemma")
      new StanfordCoreNLP(props)
    }

    def plainTextToLemmas(text: String, pipeline: StanfordCoreNLP): Seq[String] = {
      val doc = new Annotation(text)
      pipeline.annotate(doc)
      val lemmas = new ArrayBuffer[String]()
      val sentences = doc.get(classOf[SentencesAnnotation])
      for (
        sentence <- sentences;
        token <- sentence.get(classOf[TokensAnnotation])
      ) {
        val lemma = token.get(classOf[LemmaAnnotation]).toLowerCase
        if (lemma.length > 2 && !bStopWords.value.contains(lemma)
          && isOnlyLetters(lemma)) {
          lemmas += lemma
        }
      }
      lemmas
    }

    val lemmatizedMap = data.map(content => {
      val pipeline = createNLPPipeline()
      (content.getString(0), content.getString(1), content.getString(2), plainTextToLemmas(content.getString(2), pipeline))
    })

    val lemmatizedDf = lemmatizedMap.toDF("title", "genre", "plot", "features")


    //Compute an SVD decomposition of the movie plots contained in the DataFrame by using
    //the following two basic parameters:
    // numFreq = 5000 for the number of frequent terms extracted from all Wikipedia articles, and
    // k = 25 for the number of latent dimensions used for the SVD

    // select the title and features columns as an rdd
    val lemmatized = lemmatizedDf.select("title", "features").as[(String, Seq[String])].rdd

    val docTermFreqs = lemmatized.map {
      case (title, terms) => {
        val termFreqs = terms.foldLeft(new HashMap[String, Int]()) {
          (map, term) =>
          {
            map += term -> (map.getOrElse(term, 0) + 1)
            map
          }
        }
        (title, termFreqs)
      }
    }
    docTermFreqs.cache()

    val docIds = docTermFreqs.map(_._1).zipWithUniqueId().map(_.swap).collectAsMap()
    val docFreqs = docTermFreqs.map(_._2).flatMap(_.keySet).map((_, 1)).
      reduceByKey(_ + _, 24)

    val ordering = Ordering.by[(String, Int), Int](_._2)
    val topDocFreqs = docFreqs.top(numTerms)(ordering)

    val idfs = topDocFreqs.map {
      case (term, count) =>
        (term, math.log(bNumDocs.value.toDouble / count))
    }.toMap

    val idTerms = idfs.keys.zipWithIndex.toMap
    val termIds = idTerms.map(_.swap)

    val bIdfs = sc.broadcast(idfs).value
    val bIdTerms = sc.broadcast(idTerms).value

    // -------------  Prepare the Input Matrix and Compute the SVD ----------------

    val vecs = docTermFreqs.map(_._2).map(termFreqs => {
      val docTotalTerms = termFreqs.values.sum
      val termScores = termFreqs.filter {
        case (term, freq) => bIdTerms.contains(term)
      }.map {
        case (term, freq) => (bIdTerms(term), bIdfs(term) * termFreqs(term) / docTotalTerms)
      }.toSeq
      Vectors.sparse(bIdTerms.size, termScores)
    })

    vecs.cache()
    vecs.count()

    val mat = new RowMatrix(vecs)
    val svd = mat.computeSVD(k, computeU = true)


    // 1E
    // ------------------- Query the Latent Semantic Index ------------------------
    // Additionally modify the topDocsInTopConcepts function, such that it also prints the top-5 most
    //frequent genre labels of the Wikipedia articles returned by this function under each of the latent
    //concepts.

    def topTermsInTopConcepts(
                               svd: SingularValueDecomposition[RowMatrix, Matrix],
                               numConcepts: Int, numTerms: Int): Seq[Seq[(String, Double)]] = {
      val v = svd.V
      val topTerms = new ArrayBuffer[Seq[(String, Double)]]()
      val arr = v.toArray
      for (i <- 0 until numConcepts) {
        val offs = i * v.numRows
        val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex
        val sorted = termWeights.sortBy(-_._1)
        topTerms += sorted.take(numTerms).map {
          case (score, id) =>
            (bIdTerms.find(_._2 == id).getOrElse(("", -1))._1, score)
        }
      }
      topTerms
    }

    val topFrequentgenres = ArrayBuffer[LinkedHashMap[String,Int]]()

    def topDocsInTopConcepts(
                              svd: SingularValueDecomposition[RowMatrix, Matrix],
                              numConcepts: Int, numDocs: Int): Seq[Seq[(String, Double)]] = {
      val u = svd.U
      val topDocs = new ArrayBuffer[Seq[(String, Double)]]()
      for (i <- 0 until numConcepts) {
        val docWeights = u.rows.map(_.toArray(i)).zipWithUniqueId()
        topDocs += docWeights.top(numDocs).map {
          case (score, id) => (docIds(id), score)
        }
        // Algorithm to prints the top-5 most frequent genre labels of the Wikipedia articles
        val m = collection.mutable.Map[String, Int]().withDefaultValue(0)
        for ((doc_id, score) <- topDocs(topDocs.length - 1)) {
          val filtered_data = data.filter(data("title") === doc_id) // check if all titles are distinct. (They are not. I have to figure out a way)
          val genre =  filtered_data.select("genre").collect().map(_.getString(0)).mkString("")
          m(genre) = m(genre) + 1
          // get the genre to each docid and increment in the Dictionary
        }
        // pick the top 5 by value
        // add the top 5 to a variable probably declared outside the entire function
        val sorted_m = LinkedHashMap(m.toSeq.sortWith(_._2 > _._2):_*)
        val topFive = sorted_m.take(5)
        topFrequentgenres += topFive
      }
      topDocs
    }

    val topConceptTerms = topTermsInTopConcepts(svd, 25, 25)
    val topConceptDocs = topDocsInTopConcepts(svd, 25, 25)
    var i = 0
    bw.write("<----------Question 1E-------->")
    bw.write("\n")
    for ((terms, docs) <- topConceptTerms.zip(topConceptDocs)) {
      println("Concept terms: " + terms.map(_._1).mkString(", "))
      bw.write("Concept terms: " + terms.map(_._1).mkString(", "))
      bw.write("\n")

      println("Concept docs: " + docs.map(_._1).mkString(", "))
      bw.write("Concept docs: " + docs.map(_._1).mkString(", "))
      bw.write("\n")

      println("Top-5 frequent genres: " + topFrequentgenres(i))
      bw.write("Top-5 frequent genres: " + topFrequentgenres(i))
      bw.write("\n\n")
      println()
      i += 1
    }

    // 1F
    // Keyword Queries
    // Use consine similary for extracting keyword queries
    def termsToQueryVector(
                            terms: scala.collection.immutable.Seq[String],
                            idTerms: scala.collection.immutable.Map[String, Int],
                            idfs: scala.collection.immutable.Map[String, Double]): BSparseVector[Double] = {
      val indices = terms.map(idTerms(_)).toArray
      val values = terms.map(idfs(_)).toArray
      new BSparseVector[Double](indices, values, idTerms.size)
    }

    def norm(vec: Array[Double]):Double = {
      val sq = vec.map(i => i*i)
      val s = sq.sum
      val result = math.sqrt(s)
      result
    }

    def topDocsForTermQuery(
                             US: RowMatrix,
                             V: Matrix,
                             query: BSparseVector[Double]): Seq[(Double, Long)] = {
      val breezeV = new BDenseMatrix[Double](V.numRows, V.numCols, V.toArray)
      val termRowArr = (breezeV.t * query).toArray
      val normQueryVec = norm(termRowArr)
      val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)
      val cosineSimArr = US.rows.collect().map(vec => vec.toArray.zip(termRowArr).map(pair => pair._1 * pair._2).sum/(normQueryVec*norm(vec.toArray)))
      val allDocWeights =  sc.parallelize(cosineSimArr).zipWithUniqueId()
      // Docs can end up with NaN score if their row in U is all zeros. Filter these out
      allDocWeights.filter(!_._1.isNaN).top(25)
    }

    def multiplyByDiagonalRowMatrix(mat: RowMatrix, diag: MLLibVector): RowMatrix = {
      val sArr = diag.toArray
      new RowMatrix(mat.rows.map { vec =>
        val vecArr = vec.toArray
        val newArr = (0 until vec.size).toArray.map(i => vecArr(i) * sArr(i))
        Vectors.dense(newArr)
      })
    }

    val US = multiplyByDiagonalRowMatrix(svd.U, svd.s)

    bw.write("\n\n\n")
    bw.write("<---------Question 1F---------->")
    bw.write("\n\n")
    val terms = List("sweet", "love", "witch", "wife", "child")
    bw.write("The 5 interesting queries are: jealous, love, hate, desert, child")
    val queryVec = termsToQueryVector(terms, idTerms, idfs)
    val idWeights = topDocsForTermQuery(US, svd.V, queryVec)

    bw.write("\n")
    bw.write("Top-25 document vectors (and corresponding title entries): ")
    println(idWeights.map{ case (score, id) => (docIds(id), score)}.mkString(", "))
    bw.write("\n\n")
    bw.write(idWeights.map{ case (score, id) => (docIds(id), score)}.mkString(", "))
    bw.write("\n\n")

    val duration = (System.nanoTime - startTime) / 1e9d
    bw.write("Total Experiment Runtime In seconds: " + duration + "secs.")

    bw.flush()
    bw.close()
    sc.stop()
  }
}

