#pragma once

// Run parameters: the one table of every config key the code reads (params.cpp: key, type, default, section, level,
// allowed values, one-line doc) and the config-file reader built on it.
//
// - A config file sets keys of the table only: an unknown key, a value of the wrong type or outside the allowed
//   values is an error at load time (the message names the nearest valid key).
// - cfg.get_*("KEY") returns the file's value or the table default; a key not in the table, or read with another
//   type, is a programming error (abort).
// - docs/CONFIG.md is generated from the table (scripts/gen_config_md.py); config.txt holds the level-user keys.

#include <map>
#include <set>
#include <string>
#include <vector>

enum class ParamType { Bool, Int, Double, String };

// user: config.txt; expert: documented accuracy and output knobs (default = production); test: test problems;
// internal: performance tuning thresholds; experiment: open A/B switches, candidates for removal
enum class ParamLevel { User, Expert, Test, Internal, Experiment };

struct ParamSpec {
    const char* key;
    ParamType type;
    const char* def;        // default, as it would be written in the config file
    const char* section;
    ParamLevel level;
    const char* values;     // allowed values "a|b|c" (strings; "<number>" admits any number too), "" = free
    const char* doc;
};

struct DebugEnvSpec {
    const char* name;
    const char* doc;
};

const std::vector<ParamSpec>& param_table();
const std::vector<DebugEnvSpec>& debug_env_table();
const ParamSpec* find_param(const std::string& key);

struct Config {
    std::map<std::string, std::string> params;      // the file's keys (echoed into restart files and snapshots)
    std::map<std::string, std::string> overrides;   // command-line values (--test, --restart): not in params
    std::string text;                                // the file as read (echoed into restart files)
    std::string filename;

    // Reads and checks the file; exits with a message (rank 0: verbose) on an unknown key or a bad value
    void load(const std::string& file, bool verbose);
    void set_override(const std::string& key, const std::string& value);

    std::string get_str(const std::string& key) const;
    double get_double(const std::string& key) const;
    int get_int(const std::string& key) const;
    bool get_bool(const std::string& key) const;

    // Every parameter's effective value and where it came from (file / default / command line)
    void write_effective(const std::string& path) const;

    // Strict conversions (exit naming the key on failure)
    static double to_double(const std::string& key, const std::string& s);

private:
    const std::string& raw(const std::string& key, ParamType type) const;
};
