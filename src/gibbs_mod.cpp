#include <boost/python.hpp>
#include <iostream>
#include <string>
#include <vector>
#include  "gibbs_sampling_dmm.hpp"
namespace py = boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(predict_overloads, GibbsSamplingDMM::predict, 1, 3);


BOOST_PYTHON_MODULE(gibbsdmm) {
    py::class_ < GibbsSamplingDMM >("GibbsSamplingDMM", py::init<py::list, double, double, int, int, int, bool>())
    .def_pickle(gibbs_sampling_dmm_suite())
    .def("fit", &GibbsSamplingDMM::fit)
    .def("analyse_corpus", &GibbsSamplingDMM::analyse_corpus)
    .def("initialize_topic_assignments", &GibbsSamplingDMM::initialize_topic_assignments)
    .def("predict", &GibbsSamplingDMM::predict, predict_overloads(py::args("unseen_docs", "niters", "probs")))
    .add_property("topic_assignments", &GibbsSamplingDMM::topic_assignments)
    .add_property("topic_word_count", &GibbsSamplingDMM::topic_word_count)
    .add_property("sum_topic_word_count", &GibbsSamplingDMM::sum_topic_word_count)
    .add_property("id_to_word_vocabulary", &GibbsSamplingDMM::id_to_word_vocabulary)
    .add_property("topics_convergence", &GibbsSamplingDMM::topics_convergence)
    .def_readonly("vocabulary_size", &GibbsSamplingDMM::vocabulary_size)
    .def_readonly("beta", &GibbsSamplingDMM::beta)
    .def_readonly("beta_sum", &GibbsSamplingDMM::beta_sum)
    .def_readonly("top_words", &GibbsSamplingDMM::top_words)
    .def_readonly("ntopics", &GibbsSamplingDMM::ntopics);
}