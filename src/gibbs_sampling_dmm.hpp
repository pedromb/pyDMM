#include <iostream>
#include <random>
#include <string>
#include <algorithm> 
#include <vector>
#include <unordered_map>
#include <boost/python.hpp>
#include <boost/progress.hpp>

namespace py = boost::python;

using namespace std;

class GibbsSamplingDMM {
    public:
        int niters, num_documents, num_of_words_in_corpus, ntopics, top_words, vocabulary_size;;
        double alpha, alpha_sum,  beta, beta_sum;
        bool verbose;

        vector<int> _topic_assignments, _sum_topic_word_count, doc_topic_count, _topics_convergence;
        vector<double> multi_pros;
        vector<vector<int>> _topic_word_count, occurence_to_index_count, corpus;
        unordered_map<string, int> word_to_id_vocabulary;
        unordered_map<int, string> _id_to_word_vocabulary;

        py::list documents;

        GibbsSamplingDMM(py::list documents, double alpha=0.1, double beta=0.01, int ntopics=20, int niters=2000, int top_words=20, bool verbose=0) {
            this->generator();
            this->documents = documents;
            this->alpha = alpha;
            this->beta = beta;
            this->ntopics = ntopics;
            this->niters = niters;
            this->top_words = top_words;
            this->verbose = verbose;
        };

        void fit() {
            cout << "Running Gibbs sampling inference";
            boost::progress_display show_progress(this->niters);
            for(int i=1; i < this->niters+1; i++) {
                this->single_iteration();
                this->_topics_convergence.push_back(this->ntopics-count(this->_sum_topic_word_count.begin(), this->_sum_topic_word_count.end(), 0));
                ++show_progress;
            }         
        }

        void analyse_corpus() {
            this->num_of_words_in_corpus = 0;
            int index_word = -1;
            for(int i = 0; i < len(this->documents); i++){
                vector<int> current_doc, word_occurence_to_index_doc;
                unordered_map<string, int> word_occurence_to_index_doc_count;
                py::list words = py::extract<py::list>(this->documents[i]);
                for(int j = 0; j <len(words); j++) {
                    string word = py::extract<string>(words[j]);
                    if(this->word_to_id_vocabulary.find(word) == this->word_to_id_vocabulary.end()){
                        index_word += 1;
                        this->word_to_id_vocabulary[word] = index_word;
                        this->_id_to_word_vocabulary[index_word] = word;
                    }
                    current_doc.push_back(this->word_to_id_vocabulary[word]);
                    if(word_occurence_to_index_doc_count.find(word) != this->word_to_id_vocabulary.end()){
                        word_occurence_to_index_doc_count[word] += 1;
                    } else {
                        word_occurence_to_index_doc_count[word] = 1;
                    }
                    word_occurence_to_index_doc.push_back(word_occurence_to_index_doc_count[word]);
                }
                this->corpus.push_back(current_doc);
                this->num_of_words_in_corpus += len(words);
                this->occurence_to_index_count.push_back(word_occurence_to_index_doc);
            }
            this->vocabulary_size = this->word_to_id_vocabulary.size();
            this->doc_topic_count = vector<int> (this->ntopics, 0);
            this->_topic_word_count = vector<vector<int>> (this->ntopics, vector<int> (this->vocabulary_size, 0));
            this->_sum_topic_word_count = vector<int> (this->ntopics, 0);
            this->multi_pros = vector<double> (this->ntopics, 1.0/this->ntopics);
            this->num_documents = len(this->documents);
            this->beta_sum = this->vocabulary_size * this->beta;
            this->alpha_sum = this->alpha * this->ntopics;
            if(this->verbose == 1) {
                cout << "Corpus size: " << this->num_documents << " docs, " << this->num_of_words_in_corpus << " words \n";
                cout << "Vocabulary size: " << this->vocabulary_size << "\n";
                cout << "Number of topics: " << this->ntopics << "\n";
                cout << "alpha: " << this->alpha << "\n";
                cout << "beta: " << this->beta  << "\n";
                cout << "Number of sampling iterations: " << this->niters << "\n";
                cout << "Number of top topical words: " << this->top_words << "\n";
            
            
            }
        }

        void initialize_topic_assignments() {
            for(int i = 0; i < this->num_documents; i++){
                int topic = this->next_discrete(this->multi_pros);
                this->doc_topic_count[topic] += 1;
                int doc_size = this->corpus[i].size();
                for(int j = 0; j < doc_size; j++) {
                    int word = this->corpus[i][j];
                    this->_topic_word_count[topic][word] += 1;
                    this->_sum_topic_word_count[topic] += 1;
                }
                this->_topic_assignments.push_back(topic);
            }
        }

        py::list predict(py::list unseen_docs, int niters=100, bool probs=1) {

            int num_documents = len(unseen_docs), num_of_words_in_corpus = 0;

            vector<int> topic_assignments, doc_topic_count (this->ntopics, 0);
            vector<double> multi_pros (this->ntopics, 1.0/this->ntopics);
            vector<vector<int>> corpus, occurence_to_index_count;
            
            py::list results;
            

            for(int i = 0; i < num_documents; i++) {
                vector<int> current_doc, word_occurence_to_index_doc;
                unordered_map<string, int> word_occurence_to_index_doc_count;
                py::list words = py::extract<py::list>(unseen_docs[i]);
                for(int j = 0; j <len(words); j++) {
                    string word = py::extract<string>(words[j]);
                    if(this->word_to_id_vocabulary.find(word) != this->word_to_id_vocabulary.end()){
                        current_doc.push_back(this->word_to_id_vocabulary[word]);
                        if(word_occurence_to_index_doc_count.find(word) == this->word_to_id_vocabulary.end()){
                            word_occurence_to_index_doc_count[word] = 1;
                        } else {
                             word_occurence_to_index_doc_count[word] +=  1;
                        }
                        word_occurence_to_index_doc.push_back(word_occurence_to_index_doc_count[word]);
                    }
                }
                num_of_words_in_corpus += current_doc.size();
                corpus.push_back(current_doc);
                occurence_to_index_count.push_back(word_occurence_to_index_doc);
            }
            
            for(int i = 0; i < num_documents; i++){
                int topic = this->next_discrete(multi_pros);
                doc_topic_count[topic] += 1;
                int doc_size = corpus[i].size();
                for(int j = 0; j < doc_size; j++) {
                    int word = corpus[i][j];
                    this->_topic_word_count[topic][word] += 1;
                    this->_sum_topic_word_count[topic] += 1;
                }
                topic_assignments.push_back(topic);
            }

            for(int i = 0; i < niters; i++) {
                for(int d_index = 0; d_index < num_documents; d_index++) {
                    int topic = topic_assignments[d_index];
                    vector<int> document = corpus[d_index];
                    int doc_size = document.size();
                    doc_topic_count[topic] -= 1;

                    for(int w_index = 0; w_index < doc_size; w_index++) {
                        int word = document[w_index];
                        this->_topic_word_count[topic][word] -= 1;
                        this->_sum_topic_word_count[topic] -= 1;
                    }

                    for(int t_index = 0; t_index < this->ntopics; t_index++) {
                        multi_pros[t_index] = doc_topic_count[t_index] + this->alpha;
                        
                        for(int w_index = 0; w_index < doc_size; w_index++) {
                            int word = document[w_index];
                            double num = this->_topic_word_count[t_index][word] + this->beta + occurence_to_index_count[d_index][w_index] - 1;
                            double den = this->_sum_topic_word_count[t_index] + this->beta + w_index;
                            multi_pros[t_index] *= num/den;
                        }
                    }

                    topic = this->next_discrete(multi_pros);
                    doc_topic_count[topic] += 1;

                    for(int w_index = 0; w_index < doc_size; w_index++) {
                        int word = document[w_index];
                        this->_topic_word_count[topic][word] += 1;
                        this->_sum_topic_word_count[topic] += 1;
                    }

                    topic_assignments[d_index] = topic;

                }
            }

            if(probs) {

                for(int i = 0; i < num_documents; i++) {
                    py::dict probs_topics;
                    int doc_size = corpus[i].size();
                    double sum_ = 0;
                    
                    for(int t_index = 0; t_index < this->ntopics; t_index++) {
                        multi_pros[t_index] = doc_topic_count[t_index] + this->alpha;

                        for(int w_index = 0; w_index < doc_size; w_index++){
                            int word = corpus[i][w_index];
                            multi_pros[t_index] *= (this->_topic_word_count[t_index][word] + this->beta)/(this->_sum_topic_word_count[t_index] + this->beta_sum);
                        }

                        sum_ += multi_pros[t_index];
                    }

                    for(int t_index = 0; t_index < this->ntopics; t_index++) {
                        double res = multi_pros[t_index];
                        double probs_ = res / sum_;
                        probs_topics[t_index] = probs_;
                    }
                    results.append(probs_topics);
                }
                return results;
            }
            return toPythonList(topic_assignments);
        }

        py::list topic_assignments() {
            return toPythonList(this->_topic_assignments);
        }

        py::list topic_word_count() {
            return toPythonListOfList(this->_topic_word_count);
        }   

        py::list sum_topic_word_count() {
            return toPythonList(this->_sum_topic_word_count);
        }  

        py::list topics_convergence() {
            return toPythonList(this->_topics_convergence);
        }

        py::dict id_to_word_vocabulary() {
            return toPythonDict(this->_id_to_word_vocabulary);
        }  
 
        template <class K, class V>
        static py::dict toPythonDict(unordered_map<K, V> map) {
            typename unordered_map<K, V>::iterator iter;
            py::dict dictionary;
            for (iter = map.begin(); iter != map.end(); ++iter) {
                dictionary[iter->first] = iter->second;
            }
            return dictionary;
        }

        template <class T>
        static py::list toPythonList(vector<T> vector) {
            typename vector<T>::iterator iter;
            py::list list;
            for (iter = vector.begin(); iter != vector.end(); ++iter) {
                list.append(*iter);
            }
            return list;
        }

        static py::list toPythonListOfList(vector<vector<int>> vector) {
            py::list list;
            for (int i = 0; i<vector.size(); i++) {
                list.append(toPythonList(vector[i]));
            }
            return list;
        }

        static vector<int> toStdVectorInt(py::object list) {
            vector<int> res;
            for(int i = 0; i<len(list); i++) {
                res.push_back(py::extract<int>(list[i]));
            }
            return res;
        }

        static vector<vector<int>> toStdVectorOfVectorInt(py::object list) {
            vector<vector<int>> res;
            for(int i = 0; i<len(list); i++) {
                res.push_back(toStdVectorInt(list[i]));
            }
            return res;
        }

        static vector<double> toStdVectorDouble(py::object list) {
            vector<double> res;
            for(int i = 0; i<len(list); i++) {
                res.push_back(py::extract<double>(list[i]));
            }
            return res;
        }

        static unordered_map<string, int> toMapStrInt(py::object obj){
            py::dict dict = py::extract<py::dict>(obj);
            unordered_map<string, int> res;
            py::list keys = dict.keys();
            for(int i = 0; i < len(keys); i++) {
                string my_key = py::extract<string>(keys[i]);
                int value = py::extract<int>(dict[my_key]);
                res[my_key] = value;
            }
            return res;
        }

        static unordered_map<int, string> toMapIntStr(py::object obj){
            py::dict dict = py::extract<py::dict>(obj);
            unordered_map<int, string> res;
            py::list keys = dict.keys();
            for(int i = 0; i < len(keys); i++) {
                int my_key = py::extract<int>(keys[i]);
                string value = py::extract<string>(dict[my_key]);
                res[my_key] = value;
            }
            return res;
        }

    private:
        default_random_engine generator;
        
        int next_discrete(vector<double> probs) {
            uniform_real_distribution<double> distribution(0.0, 1.0);
            double sum_ = 0;
            for(int i = 0; i < probs.size(); i++) {
                sum_ += probs[i];
            }
            double r_unif = distribution(this->generator);
            double r = sum_*r_unif;
            sum_ = 0;
            for(int i = 0; i <probs.size(); i++) {
                sum_ += probs[i];
                if(sum_ > r) {
                    return(i);
                }
            }
            return (probs.size()-1);
        }

        void single_iteration() {
            for(int d_index = 0; d_index < this->num_documents; d_index++){
                int topic = this->_topic_assignments[d_index];
                vector<int> document = this->corpus[d_index];
                int doc_size = document.size();

                this->doc_topic_count[topic] -= 1;

                for(int w_index = 0; w_index < doc_size; w_index++){
                    int word = document[w_index];
                    this->_topic_word_count[topic][word] -= 1;
                    this->_sum_topic_word_count[topic] -= 1;
                }

                for(int t_index = 0; t_index <- this->ntopics; t_index++){
                    
                    this->multi_pros[t_index] = this->doc_topic_count[t_index] + this->alpha;
                    
                    for(int w_index = 0; w_index < doc_size; w_index++){
                        int word = document[w_index];
                        double num = this->_topic_word_count[t_index][word] + this->beta + this->occurence_to_index_count[d_index][w_index] - 1;
                        double den = this->_sum_topic_word_count[t_index] + this->beta_sum + w_index;
                        this->multi_pros[t_index] *= num/den;
                    }

                }

                topic = this->next_discrete(this->multi_pros);
                this->doc_topic_count[topic] += 1;

                for(int w_index = 0; w_index < doc_size; w_index++){
                    int word = document[w_index];
                    this->_topic_word_count[topic][word] += 1;
                    this->_sum_topic_word_count[topic] += 1;
                }
                
                this->_topic_assignments[d_index] = topic;

            }
        }

};

struct gibbs_sampling_dmm_suite : py::pickle_suite {
    static py::tuple getinitargs(const GibbsSamplingDMM& g) {
        return py::make_tuple(g.documents, g.alpha, g.beta, g.ntopics, g.niters, g.top_words, g.verbose);
    }

    static py::tuple getstate(const GibbsSamplingDMM& g) {
        
        py::list py_topic_assignments = g.toPythonList(g._topic_assignments), 
            py_sum_topic_word_count = g.toPythonList(g._sum_topic_word_count),
            py_doc_topic_count = g.toPythonList(g.doc_topic_count), py_topics_convergence = g.toPythonList(g._topics_convergence), 
            py_multi_pros = g.toPythonList(g.multi_pros), py_topic_word_count = g.toPythonListOfList(g._topic_word_count),
            py_occurence_to_index_count = g.toPythonListOfList(g.occurence_to_index_count),  
            py_corpus = g.toPythonListOfList(g.corpus);
        
        py::dict py_word_to_id_vocabulary = g.toPythonDict(g.word_to_id_vocabulary), 
            py_id_to_word_vocabulary = g.toPythonDict(g._id_to_word_vocabulary);

        return py::make_tuple(g.num_documents, g.num_of_words_in_corpus, g.vocabulary_size, g.alpha_sum, g.beta_sum, 
            py_topic_assignments, py_sum_topic_word_count, py_doc_topic_count, py_topics_convergence, py_multi_pros, 
            py_topic_word_count, py_occurence_to_index_count, py_corpus, py_word_to_id_vocabulary, py_id_to_word_vocabulary);
    }

    static void setstate(GibbsSamplingDMM& g, py::tuple state) {
        g.num_documents = py::extract<int>(state[0]);
        g.num_of_words_in_corpus = py::extract<int>(state[1]);
        g.vocabulary_size = py::extract<int>(state[2]);
        g.alpha_sum = py::extract<double>(state[3]);
        g.beta_sum = py::extract<double>(state[4]);
        g._topic_assignments = g.toStdVectorInt(state[5]);
        g._sum_topic_word_count = g.toStdVectorInt(state[6]);
        g.doc_topic_count = g.toStdVectorInt(state[7]);
        g._topics_convergence = g.toStdVectorInt(state[8]);
        g.multi_pros = g.toStdVectorDouble(state[9]);
        g._topic_word_count = g.toStdVectorOfVectorInt(state[10]);
        g.occurence_to_index_count = g.toStdVectorOfVectorInt(state[11]);
        g.corpus = g.toStdVectorOfVectorInt(state[12]);
        g.word_to_id_vocabulary = g.toMapStrInt(state[13]);
        g._id_to_word_vocabulary = g.toMapIntStr(state[14]);
    }
};