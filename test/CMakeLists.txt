include(GoogleTest)
find_package(GTest CONFIG REQUIRED)
add_executable(PokerTest
	poker_tests.cpp
	execution_tests.cpp
	game_test.cpp
)
target_link_libraries(PokerTest gtest::gtest GTest::gtest_main bshoshany-thread-pool::bshoshany-thread-pool)
add_test(NAME PokerTest COMMAND PokerTest)
gtest_discover_tests(PokerTest)