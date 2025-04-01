#include "bm_cli.hpp"

#include <iomanip>
#include <iostream>
#include <ostream>
#include <regex>
#include <vector>

namespace Colors {
  inline constexpr std::string_view Reset     = "\033[0m";
  inline constexpr std::string_view Usage     = "\033[0m\033[1;37m";
  inline constexpr std::string_view Box       = "\033[2;90m";
  inline constexpr std::string_view BoxHeader = "\033[1;90m";
  inline constexpr std::string_view Header    = "\033[1m\033[4;37m";
  inline constexpr std::string_view Info      = "\033[1;37m";
  inline constexpr std::string_view ShortOpt  = "\033[1;32m";
  inline constexpr std::string_view LongOpt   = "\033[1;36m";
  inline constexpr std::string_view Args      = "\033[1;33m";
  inline constexpr std::string_view Descript  = "\033[0m";
  inline constexpr std::string_view Comment   = "\033[90m";
}

namespace BoxDraw {
  inline const std::string Vertical     = std::string(Colors::Reset) + std::string(Colors::Box) + "│" + std::string(Colors::Reset);
  inline const std::string Horizontal   = std::string(Colors::Reset) + std::string(Colors::Box) + "─" + std::string(Colors::Reset);
  inline const std::string BottomLeft   = std::string(Colors::Reset) + std::string(Colors::Box) + "╰" + std::string(Colors::Reset);
  inline const std::string BottomRight  = std::string(Colors::Reset) + std::string(Colors::Box) + "╯" + std::string(Colors::Reset);
  inline const std::string TopLeft      = std::string(Colors::Reset) + std::string(Colors::Box) + "╭" + std::string(Colors::Reset);
  inline const std::string TopRight     = std::string(Colors::Reset) + std::string(Colors::Box) + "╮" + std::string(Colors::Reset);
}

inline constexpr int small_spacing = 1;
inline constexpr int spacing = 2; // minimum spacing between columns
inline constexpr int long_alignment = 2 * spacing; // alignemnt start for options (not headers, not info, no short flag)

/* Type alias for a tuple representing a line of option information */
using OptionLine = std::tuple<std::string, std::string, std::string, std::string, std::string, std::string, std::string>;

/**
 * @brief Computes the visible length of a string ignoring ANSI escape codes
 * 
 * @param str The input string
 * @return int
 */
static int get_visible_length(const std::string& str) {
  static const std::regex ansi_escape("\033\\[[0-9;]*m");
  return std::regex_replace(str, ansi_escape, "").length();
}

/**
 * @brief Formats the requirement string for an option
 * 
 * @param option The required option
 * @param argument The required argument(s)
 * @param base_format The base format (what should be used for the rest of the string)
 * @return std::string The formatted rquirement string
 */
static std::string format_requirement(
  const std::string& option,
  const std::string& argument,
  const std::string& base_format=""
) {
  return std::string(Colors::Reset) + base_format + "(requires " + std::string(Colors::LongOpt) + option + base_format + " " + std::string(Colors::Args) + argument + base_format + ")" + std::string(Colors::Reset);
}

inline const std::vector<OptionLine> option_lines = {
  { "Help:", "", "", "", "", "", ""                                                                                                           },
  { "", "", "-h", "--help",           "",                 "show this help message", ""                                                        },
  { "Benchmark options:", "", "", "", "", "", ""                                                                                              },
  { "", "", "-f", "--file",           "<dir_name>",       "specify output CSV file", "(inside /results/raw/)"                                 },
  { "", "", "",   "",                 "",                 "will be created if it doesn't exist", ""                                           },
  { "", "", "-a", "--append",         "",                 "append results to the output file", "(default: overwrite)"                         },
  { "", "", "-i", "--iterations",     "<num>",            "number of benchmark iterations", "(default 10)"                                    },
  { "Algorithm selection:", "", "", "", "", "", ""                                                                                            },
  { "", "If no algorithm is selected, all algorithms will be run", "", "", "", "", ""                                                         },
  { "", "", "",   "--base",           "cm256,isal,leopard,wirehair",     "run the specified algorithms,", "0 or more comma seperated args"    },
  { "", "", "",   "--xorec",          "cpu,unified-ptr,gpu-ptr,gpu-cmp", "run the selected XOR-EC variants", "0 or more comma seperated args" },
  { "", "", "",   "",                 "",                 "","cpu:          data buffer, parity buffer and"                                   },
  { "", "", "",   "",                 "",                 "","              computation on CPU"                                               },
  { "", "", "",   "",                 "",                 "","unified-ptr:  data buffer in unified memory, parity"                            },
  { "", "", "",   "",                 "",                 "","              buffer and computation on CPU"                                    },
  { "", "", "",   "",                 "",                 "","gpu-ptr:      data buffer on GPU, parity buffer and"                            },
  { "", "", "",   "",                 "",                 "","              computation on CPU"                                               },
  { "", "", "",   "",                 "",                 "","gpu-cmp:      data buffer, parity buffer and"                                   },
  { "", "", "",   "",                 "",                 "","              computation on GPU"                                               },
  { "XOR-EC version selection:", "", "", "", "", "", ""                                                                                       },
  { "", "", "",   "--simd",           "scalar,avx,avx2,avx512", "which SIMD version to use for XOR-EC benchmarking", ""                       },
  { "", "", "",   "",                 "",                 format_requirement("--xorec", "scalar|unified-ptr|gpu-ptr"), ""                     },
  { "", "", "",   "",                 "",                 "","0 or more comma separated args (default: all)"                                  }
};

void print_usage() {
  std::cout << Colors::Reset << "\nUsage: " << Colors::Usage << " ec-benchmark [OPTIONS(0)...] [ : [OPTIONS(N)...]]" << Colors::Reset << "\n\n";
}

void print_options() {
  int max_width = 0;
  int max_left_width = 0;
  
  for (const auto& [header, info, short_opt, long_opt, args, description, comment] : option_lines) {
    int left_width  = spacing
                    + get_visible_length(short_opt) + spacing
                    + get_visible_length(long_opt) + spacing
                    + get_visible_length(args) + small_spacing;

    int total_width = get_visible_length(header)
                    + get_visible_length(info)
                    + left_width
                    + get_visible_length(description) + small_spacing
                    + get_visible_length(comment) + spacing;

    max_left_width = std::max(max_left_width, left_width);
    max_width = std::max(max_width, total_width);
  }


  // Build top and bottom box borders.
  std::string box_top = std::string(BoxDraw::TopLeft)
                      + std::string(BoxDraw::Horizontal)
                      + std::string(Colors::BoxHeader) + " Options " + std::string(Colors::Reset);
  for (int i = 11; i <= max_width + spacing; ++i) { // i starts at the index after the word "Options"
    box_top += std::string(BoxDraw::Horizontal);
  }
  box_top += std::string(BoxDraw::TopRight);

  std::string box_bottom = std::string(BoxDraw::BottomLeft);
  for (int i = 1; i <= max_width + spacing; ++i) { // i starts at the index after the bottom left corner's character
    box_bottom += std::string(BoxDraw::Horizontal);
  }
  box_bottom += std::string(BoxDraw::BottomRight);

  // Print the top border
  std::cout << box_top << '\n';

  for (const auto& [header, info, short_opt, long_opt, args, descript, comment] : option_lines) {
    int len_header    = get_visible_length(header);
    int len_info      = get_visible_length(info);
    int len_short_opt = get_visible_length(short_opt);
    int len_long_opt  = get_visible_length(long_opt);
    int len_args      = get_visible_length(args);
    int len_descript  = get_visible_length(descript);
    int len_comment   = get_visible_length(comment);
    int curr_len      = 0;

    // Print a blank line if the current line is a header
    if (len_header > 0) {
      std::cout << BoxDraw::Vertical
                << std::setw(max_width + spacing) << std::setfill(' ') << std::string(spacing, ' ')
                << BoxDraw::Vertical << '\n';
    }

    std::cout << BoxDraw::Vertical << std::string(spacing, ' ');

    if (len_header > 0) {
      std::cout << Colors::Header << header << Colors::Reset;
      curr_len += len_header;
    }

    if (len_info > 0) {
      std::cout << Colors::Info << info << Colors::Reset;
      curr_len += len_info;
    }

    if (len_short_opt > 0) {
      std::cout << Colors::ShortOpt << short_opt << Colors::Reset;
      curr_len += len_short_opt;
    }
    
    if (len_long_opt > 0) {
      std::cout << std::setw(long_alignment-curr_len) << std::setfill(' ') << std::string(spacing, ' ');
      std::cout << Colors::LongOpt << long_opt << Colors::Reset;
      curr_len = long_alignment + len_long_opt;
    }

    if (len_args > 0) {
      std::cout << std::string(small_spacing, ' ');
      curr_len += small_spacing;
      std::cout << Colors::Args << args << Colors::Reset;
      curr_len += len_args;
    }

    if (len_descript > 0) {
      std::cout << std::setw(max_left_width - curr_len) << std::setfill(' ') << std::string(spacing, ' ');
      std::cout << Colors::Descript << descript << Colors::Reset;
      curr_len = max_left_width + len_descript;
    }

    if (len_comment > 0) {
      if (max_left_width > curr_len) { // There was no description
        std::cout << std::setw(max_left_width - curr_len) << std::setfill(' ') << std::string(spacing, ' ')
                  << Colors::Comment << comment << Colors::Reset;
        curr_len = max_left_width + len_comment;
      } else {
        std::cout << std::string(small_spacing, ' ')
                  << Colors::Comment << comment << Colors::Reset;
        curr_len += small_spacing + len_comment;
      }
    }

    std::cout << std::setw(max_width-curr_len) << std::setfill(' ') << std::string(spacing, ' ')
              << BoxDraw::Vertical << '\n';
  }

  std::cout << box_bottom << '\n';
}
