include(GoogleTest)
find_package(GTest CONFIG REQUIRED)
add_executable(PokerTest
	poker_tests.cpp
	execution_tests.cpp
	game_test.cpp
	game_logic_test.cpp
	mccfr_test.cpp
	hunl_test.cpp
	hand_indexer_test.cpp
	card_abstraction_test.cpp
	push_fold_test.cpp
	lbr_test.cpp
	river_solver_test.cpp
	river_showdown_test.cpp
)
target_compile_definitions(PokerTest PRIVATE POKER_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
target_link_libraries(PokerTest gtest::gtest GTest::gtest_main bshoshany-thread-pool::bshoshany-thread-pool)
add_test(NAME PokerTest COMMAND PokerTest)
gtest_discover_tests(PokerTest)
