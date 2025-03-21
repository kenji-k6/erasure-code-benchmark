#include "bm_cli.hpp"
#include <iostream>
#include <ostream>
#include <iomanip>
#include <vector>
#include <regex>

#define CYAN << "\033[1;94m"


#define Format const std::string
#define BoxDraw const std::string

Format USAGE = "\033[0m\033[1;37m";
Format RST = "\033[0m"; // Reset format
Format BOX = "\033[2;90m";
Format BOXHEADER = "\033[1;90m";

Format HEADER = "\033[1m\033[4;37m";
Format INFO =  "\033[1;37m";
Format SHORT = "\033[1;32m";
Format LONG = "\033[1;36m";
Format ARGS  = "\033[1;33m";
Format DESCRIPT = "\033[0m";
Format COMMENT = "\033[90m";

BoxDraw v   = RST + BOX + "│" + RST;
BoxDraw h   = RST + BOX + "─" + RST;
BoxDraw bl  = RST + BOX + "╰" + RST;
BoxDraw br  = RST + BOX + "╯" + RST;
BoxDraw tl  = RST + BOX + "╭" + RST;
BoxDraw tr  = RST + BOX + "╮" + RST;

BoxDraw box_header = RST + BOXHEADER + " Options " + RST;
BoxDraw box_start_top = tl + h;
BoxDraw box_start_bot = bl;

constexpr int w_small_spacing = 1;
constexpr int w_spacing = 2; //minimum spacing between columns
const int w_box_start_top = 2;
const int w_box_start_bot = 1;
const int w_box_header = 9;


void print_usage() {
  std::cout << RST << "\nUsage: " << USAGE << " ec-benchmark [OPTIONS(0)...] [ : [OPTIONS(N)...]]" << RST << "\n\n";
}


using LineTuple = std::tuple<std::string, std::string, std::string, std::string, std::string, std::string, std::string>;

static std::string get_require(std::string opt, std::string arg, std::string default_fmt=RST) {
  return default_fmt + "(requires " + LONG + opt + default_fmt + " " + ARGS + arg + default_fmt + ")";
}


static int get_visible_length(const std::string& inp) {
  static const std::regex ansi_escape("\033\\[[0-9;]*m");
  return std::regex_replace(inp, ansi_escape, "").length();
}

const std::vector<LineTuple> lines = {                                                                                   
  { "Help:", "", "", "", "", "", ""                                                                                             },
  { "", "", "-h", "--help",           "",                 "show this help message", ""                                          },
  { "Benchmark options:", "", "", "", "", "", ""                                                                                },
  { "", "", "-r", "--result-dir",     "<dir_name>",       "specify output result subdirectory", "(inside /results/raw/)"        },
  { "", "", "",   "",                 "",                 "will be created if it doesn't exist", ""                             },
  { "", "", "-a", "--append",         "",                 "append results to the output file", "(default: overwrite)"           },
  { "", "", "-b", "--benchmark",      "ec,perf",          "specify the type of benchmark to run,", "1 or 2 comma"                },
  { "", "", "",   "",                 "",                 "", "separated args (default: both)"                                  },
  { "", "", "-i", "--iterations",     "<num>",            "number of benchmark iterations", "(default 10)"                      },
  { "Algorithm selection:", "", "", "", "", "", ""                                                                              },
  { "", "If no algorithm is selected, all algorithms will be run", "", "", "", "", ""                                           },
  { "", "", "",   "--base",           "cm256,isal,leopard,wirehair", "run the specified algorithms,", "0 or more comma seperated args"      },
  { "", "", "",   "--xorec",          "cpu,unified-ptr,gpu-ptr,gpu-cmp",                 "run the selected XOR-EC variants", "0 or more comma seperated args"  },
  { "", "", "",   "",                 "",                 "","cpu:          data buffer, parity buffer and"                     },
  { "", "", "",   "",                 "",                 "","              computation on CPU"                                 },
  { "", "", "",   "",                 "",                 "","unified-ptr:  data buffer in unified memory, parity"              },
  { "", "", "",   "",                 "",                 "","              buffer and computation on CPU"                      },
  { "", "", "",   "",                 "",                 "","gpu-ptr:      data buffer on GPU, parity buffer and"              },
  { "", "", "",   "",                 "",                 "","              computation on CPU"                                 },
  { "", "", "",   "",                 "",                 "","gpu-cmp:      data buffer, parity buffer and"                     },
  { "", "", "",   "",                 "",                 "","              computation on GPU"                                 },
  { "XOR-EC version selection:", "", "", "", "", "", ""                                                                         },
  { "", "", "",   "--simd",           "scalar,avx,avx2,avx512", "which SIMD version to use for XOR-EC benchmarking", ""               },
  { "", "", "",   "",                 "",                 get_require("--xorec", "scalar|unified-ptr|gpu-ptr"), ""              },
  { "", "", "",   "",                 "",                 "","0 or more comma separated args (default: all)"                    },
  {"XOR-EC GPU Options"+RST+" "+get_require("--xorec","unified-ptr"),  "", "", "", "", "", ""                                   },
  { "", "", "",   "--touch-unified",  "true,false",       "whether to touch unified memory on the GPU before encoding,", ""      },
  { "", "", "",   "",                 "",                 "", "1 or 2 comma separated args (default: false)"                    },
  { "", "", "",   "--prefetch",       "true,false",       "whether data blocks from unified memory, are prefetched or", ""      },
  { "", "", "",   "",                 "",                 "fetched on-demand,", "1 or 2 comma separated args (default: false)"  },
  { "", "", "",   "--perf-xorec",     "scalar,avx,avx2,avx512", "", ""                                                          },
  { "", "", "",   "",                 "",                 "", "0 or more comma seperated args (default: none)"                  }
};


void print_options() {

  //<< std::setw(2) << std::setfill('0') << hours << ':'
  // std::cout << std::;
  int w_max = 0;
  int w_left_max = 0;
  for ([[maybe_unused]] auto [header, info, short_opt, long_opt, args, descript, comment] : lines) {
    int w_left  = w_spacing
                + get_visible_length(short_opt) + w_spacing
                + get_visible_length(long_opt) + w_spacing
                + get_visible_length(args) + w_small_spacing;
    int width = get_visible_length(header) + get_visible_length(info)+ w_left + get_visible_length(descript) + w_small_spacing + get_visible_length(comment) + w_spacing;
      
    w_max = std::max(w_max, width);
    w_left_max = std::max(w_left_max, w_left);
  }

  int long_alignment = 2*w_spacing;
  
  std::string box_top = box_start_top + box_header;
  std::string box_bot = box_start_bot;
  for (int i = w_box_start_top+w_box_header; i <= w_max+w_spacing; ++i) {
    box_top += h;
  }
  
  for (int i = w_box_start_bot; i <= w_max+w_spacing; ++i) {
    box_bot += h;
  }
  box_top += tr;
  box_bot += br;
  std::cout << box_top << '\n';

  for (auto [header, info, short_opt, long_opt, args, descript, comment] : lines) {
    int w_header = get_visible_length(header);
    int w_info = get_visible_length(info);
    int w_short_opt = get_visible_length(short_opt);
    int w_long_opt = get_visible_length(long_opt);
    int w_args = get_visible_length(args);
    int w_descript = get_visible_length(descript);
    int w_comment = get_visible_length(comment);


    if (w_header > 0) {
      std::cout << v << std::setw(w_max+w_spacing) << std::setfill(' ') << std::string(w_spacing, ' ') << v << '\n';
    }
    std::cout << v << std::string(w_spacing, ' ');
    int curr_width = 0;
    if (w_header > 0) {
      std::cout << HEADER << header << RST;
      curr_width += w_header;
    }

    if (w_info > 0) {
      std::cout << INFO << info << RST;
      curr_width += w_info;
    }

    if (w_short_opt > 0) {
      std::cout << SHORT << short_opt << RST;
      curr_width += w_short_opt;
    }
    
    if (w_long_opt > 0) {
      // align
      std::cout << std::setw(long_alignment-curr_width) << std::setfill(' ') << std::string(w_spacing, ' ');
      std::cout << LONG << long_opt << RST;
      curr_width = long_alignment + w_long_opt;
    }

    if (w_args > 0) {
      std::cout << std::string(w_small_spacing, ' ');
      curr_width += w_small_spacing;
      std::cout << ARGS << args << RST;
      curr_width += w_args;
    }

    if (w_descript > 0) {
      std::cout << std::setw(w_left_max - curr_width) << std::setfill(' ') << std::string(w_spacing, ' ');
      std::cout << DESCRIPT << descript << RST;
      curr_width = w_left_max + w_descript;
    }

    if (w_comment > 0) {
      if (w_left_max > curr_width) {
        std::cout << std::setw(w_left_max - curr_width) << std::setfill(' ') << std::string(w_spacing, ' ')
                  << COMMENT << comment << RST;
        curr_width = w_left_max + w_comment;
      } else {
        std::cout << std::string(w_small_spacing, ' ')
                  << COMMENT << comment << RST;
        curr_width += w_small_spacing + w_comment;
      }
      
    }

    std::cout << std::setw(w_max-curr_width) << std::setfill(' ') << std::string(w_spacing, ' ')
              << v << '\n';
  }

  std::cout << box_bot << '\n';
}
