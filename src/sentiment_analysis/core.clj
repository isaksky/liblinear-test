(ns sentiment-analysis.core
  (:import java.io.File))

(use '[clj-liblinear.core :only [train predict]]
     '[clojure.string :only [split lower-case]]
     '[clojure.java.io :only [reader]])

(defn combine-paths [path1 path2]
  (let [file1 (File. path1)
        file2 (File. file1 path2)]
    (. file2 getPath)))

(def working-dir-path
  (. System getProperty "user.dir"))

(def subjective-data-path
  (combine-paths working-dir-path "resources/rotten_imdb/quote.tok.gt9.5000"))

(def objective-data-path
  (combine-paths working-dir-path "resources/rotten_imdb/plot.tok.gt9.5000"))

(defn line-arys [path]
  (with-open [rdr (reader path)]
    (doall (map #(set (split % #" ")) (line-seq rdr)))))

(def training-cutoff 3/4) ;; what percentage of data to train on?

(let [subj-data (line-arys subjective-data-path)
      obj-data (line-arys objective-data-path)
      num-to-train-on (int (* training-cutoff (count obj-data)))
      num-to-test-with (- (count obj-data) num-to-train-on)]

  (def subjective?-model
    (train (mapcat #(take num-to-train-on %) [subj-data obj-data])
           (mapcat #(take num-to-train-on (repeat %)) [1 0])))

  ;; for REPL testing
  (defn subjective? [s] (predict subjective?-model
                                 (into #{} (split s #" "))))

  ;; test this model with the rest of the data
  (println "subjective/objective test (0 is objective, 1 is subjective)")
  (println (zipmap [:subjective :objective]
                   (map (fn [as]
                          (frequencies (map #(predict subjective?-model %) as)))
                        (map #(drop num-to-train-on %) [subj-data obj-data])))))


(let [neg-files (filter (fn [f] (.endsWith (.getPath f) "txt"))
                        (file-seq (clojure.java.io/file
                                   (combine-paths working-dir-path
                                                  "resources/review_polarity/txt_sentoken/neg"))))
      negs (remove empty? (mapcat line-arys neg-files))
      pos-files (filter (fn [f] (.endsWith (.getPath f) "txt"))
                        (file-seq (clojure.java.io/file
                                   (combine-paths working-dir-path
                                                  "resources/review_polarity/txt_sentoken/pos"))))
      poss (remove empty? (mapcat line-arys pos-files))
      neg-num-to-train-on (int (* training-cutoff (count negs)))
      pos-num-to-train-on (int (* training-cutoff (count poss)))]

  (def polarity-model (train (concat (take neg-num-to-train-on negs)
                                     (take pos-num-to-train-on poss))
                             (concat  (take neg-num-to-train-on (repeat 0))
                                      (take pos-num-to-train-on (repeat 1)))))

  ;; for REPL testing
  (defn positive? [s] (predict polarity-model
                               (into #{} (split s #" "))))

  ;; test this model with the rest of the data
  (println "negative/positive test (0 is negative, 1 is positive)")
  (println {:negative (frequencies (map #(try (predict polarity-model %) (catch Exception ex ex))
                                        (drop neg-num-to-train-on negs)))
            :positive (frequencies (map #(try (predict polarity-model %) (catch Exception ex ex))
                                        (drop pos-num-to-train-on poss)))}))
