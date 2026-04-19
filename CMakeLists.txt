cmake_minimum_required(VERSION 3.16)
project(VishingFraudDetector LANGUAGES NONE)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(SAMPLES_DIR ${CMAKE_SOURCE_DIR}/vishing/samples)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/vishing/test)

# --- Install dependencies ---
add_custom_target(install_deps
    COMMAND ${Python3_EXECUTABLE} -m pip install -r ${CMAKE_SOURCE_DIR}/requirements.txt
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Installing Python dependencies"
)

# --- Tune threshold on dev samples ---
add_custom_target(tune
    COMMAND ${Python3_EXECUTABLE} ${SRC_DIR}/tune_threshold.py
        --samples_dir ${SAMPLES_DIR}
        --output ${CMAKE_BINARY_DIR}/best_config.json
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Tuning ensemble weights on dev samples"
)

# --- Run detector on test/Fraud ---
add_custom_target(test_fraud
    COMMAND ${Python3_EXECUTABLE} ${SRC_DIR}/run_test.py
        --folder ${TEST_DIR}/Fraud
        --output ${CMAKE_BINARY_DIR}/results_fraud.csv
        --verbose
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running detector on test/Fraud"
)

# --- Run detector on test/NotFraud ---
add_custom_target(test_legit
    COMMAND ${Python3_EXECUTABLE} ${SRC_DIR}/run_test.py
        --folder ${TEST_DIR}/NotFraud
        --output ${CMAKE_BINARY_DIR}/results_legit.csv
        --verbose
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running detector on test/NotFraud"
)

# --- Run detector on all test files ---
add_custom_target(test_all
    DEPENDS test_fraud test_legit
    COMMENT "Running detector on all test files"
)

# --- Run on dev samples (sanity check) ---
add_custom_target(test_samples
    COMMAND ${Python3_EXECUTABLE} ${SRC_DIR}/run_test.py
        --folder ${SAMPLES_DIR}/Fraud
        --output ${CMAKE_BINARY_DIR}/results_samples_fraud.csv
        --verbose
    COMMAND ${Python3_EXECUTABLE} ${SRC_DIR}/run_test.py
        --folder ${SAMPLES_DIR}/NotFraud
        --output ${CMAKE_BINARY_DIR}/results_samples_legit.csv
        --verbose
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Sanity check on dev samples"
)

message(STATUS "")
message(STATUS "Available targets:")
message(STATUS "  make install_deps   -- pip install requirements")
message(STATUS "  make tune           -- tune weights on vishing/samples")
message(STATUS "  make test_all       -- run detector on all test files")
message(STATUS "  make test_fraud     -- run detector on test/Fraud only")
message(STATUS "  make test_legit     -- run detector on test/NotFraud only")
message(STATUS "  make test_samples   -- sanity check on dev samples")
message(STATUS "")
