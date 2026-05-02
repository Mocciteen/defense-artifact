# Portable out-of-tree Intel Pin tool build.
#
# Usage:
#   make PIN_ROOT=/path/to/pin
#   make PIN_ROOT=/path/to/pin obj-intel64/paddrtrace.so

PIN_ROOT ?= /path/to/pin
CONFIG_ROOT := $(PIN_ROOT)/source/tools/Config

include $(CONFIG_ROOT)/makefile.config
include makefile.rules
include $(TOOLS_ROOT)/Config/makefile.default.rules
