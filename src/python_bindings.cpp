#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/deck.hpp"
#include "../include/game.hpp"

namespace py = pybind11;

PYBIND11_MODULE(poker, m)
{
    m.doc() = "Poker game simulation module";
    py::class_<pcg64>(m, "pcg64").def(py::init<>()).def(py::init<std::uint64_t>(), py::arg("seed"));
    py::enum_<Rank>(m, "Rank")
        .value("Two", Rank::Two)
        .value("Three", Rank::Three)
        .value("Four", Rank::Four)
        .value("Five", Rank::Five)
        .value("Six", Rank::Six)
        .value("Seven", Rank::Seven)
        .value("Eight", Rank::Eight)
        .value("Nine", Rank::Nine)
        .value("Ten", Rank::Ten)
        .value("Jack", Rank::Jack)
        .value("Queen", Rank::Queen)
        .value("King", Rank::King)
        .value("Ace", Rank::Ace)
        .export_values()
        .def("__repr__", [](const Rank &r) {
            std::ostringstream oss;
            oss << r;
            return oss.str();
        });
    py::enum_<Suit>(m, "Suit")
        .value("Clubs", Suit::Clubs)
        .value("Diamonds", Suit::Diamonds)
        .value("Hearts", Suit::Hearts)
        .value("Spades", Suit::Spades)
        .export_values()
        .def("__repr__", [](const Suit &s) {
            std::ostringstream oss;
            oss << s;
            return oss.str();
        });
    py::class_<Card>(m, "Card")
        .def(py::init<Suit, Rank>())
        .def_property_readonly("rank", &Card::getRank)
        .def_property_readonly("suit", &Card::getSuit)
        .def_static("parse_card", &Card::parseCard, py::arg("card"))
        .def("__eq__", &Card::operator==)
        .def("__ne__", [](const Card c1, const Card c2) {
            return !(c1 == c2);
        })
        .def("__repr__", [](const Card c) {
            std::ostringstream oss;
            oss << c;
            return oss.str();
        });
    py::class_<Deck>(m, "Deck")
        .def(pybind11::init<>())
        .def("add_card", &Deck::addCard, py::arg("card"))
        .def("remove_cards", &Deck::removeCards, py::arg("cards"))
        .def("pop_card", &Deck::popCard)
        .def("pop_random_cards", &Deck::popRandomCards, py::arg("generator"), py::arg("count"))
        .def("at", &Deck::at, py::arg("index"))
        .def_static("create_full_deck", &Deck::createFullDeck)
        .def_static("create_deck", py::overload_cast<const std::initializer_list<Card>>(&Deck::createDeck), py::arg("cards"))
        .def_static("create_deck", py::overload_cast<const std::initializer_list<Deck>>(&Deck::createDeck), py::arg("decks"))
        .def_static("empty_deck", &Deck::emptyDeck)
        .def_static("parse_hand", &Deck::parseHand, py::arg("hand"))
        .def_property_readonly("size", &Deck::size)
        .def_property_readonly("mask", &Deck::getMask)
        .def("__repr__", [](const Deck &d) {
            std::ostringstream oss;
            oss << d;
            return oss.str();
        });
    py::class_<BS::thread_pool<BS::tp::none>>(m, "ThreadPool")
        .def(py::init<std::size_t>(), py::arg("num_threads") = std::thread::hardware_concurrency());
    m.def("player_wins_random_game", &playerWinsRandomGame, py::arg("rng"), py::arg("player_cards"), py::arg("table_cards"), py::arg("num_players"));
    m.def("probability_of_winning", py::overload_cast<pcg64 &, const Deck, const Deck, std::size_t, std::size_t>(&probabilityOfWinning), py::arg("rng"), py::arg("player_cards"), py::arg("table_cards"), py::arg("num_simulations"), py::arg("num_players"));
    m.def("probability_of_winning", py::overload_cast<const Deck, const Deck, std::size_t, std::size_t, std::size_t>(&probabilityOfWinning), py::arg("player_cards"), py::arg("table_cards"), py::arg("num_simulations"), py::arg("num_players"), py::arg("num_threads") = std::thread::hardware_concurrency());
    m.def("probability_of_winning", py::overload_cast<const Deck, const Deck, std::size_t, std::size_t, BS::thread_pool<BS::tp::none> &>(&probabilityOfWinning), py::arg("player_cards"), py::arg("table_cards"), py::arg("num_simulations"), py::arg("num_players"), py::arg("thread_pool"));
}