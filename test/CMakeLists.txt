include(GoogleTest)
find_package(GTest CONFIG REQUIRED)
add_executable(PokerTest
	poker_tests.cpp
	execution_tests.cpp
	game_test.cpp
	game_logic_test.cpp
)
target_link_libraries(PokerTest gtest::gtest GTest::gtest_main bshoshany-thread-pool::bshoshany-thread-pool)
add_test(NAME PokerTest COMMAND PokerTest)
gtest_discover_tests(PokerTest)

add_executable(PokerMLTest
	ml_test.cpp
)
target_link_libraries(PokerMLTest
	gtest::gtest GTest::gtest_main
	bshoshany-thread-pool::bshoshany-thread-pool
	dlib::dlib
)
add_test(NAME PokerMLTest COMMAND PokerMLTest)
gtest_discover_tests(PokerMLTest)
