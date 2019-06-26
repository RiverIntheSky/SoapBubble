enable_testing()

include_directories(/usr/include/gtest)

set(test_src
  run_all.cc
  sample_test.cc
  kamino.cc
  )

add_executable(kamino_test ${test_src})

target_link_libraries(kamino_test
  gtest)

add_test(
  NAME tests
  COMMAND kamino_test
  )
