# ----------------------------------------------------------------------------
# Copyright (c) 2018 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    import os
    from argparse import ArgumentParser

    from mako.template import Template

    desc = (
        'Generate Digital RF Source/Sink GRC block xml file from template.'
    )
    parser = ArgumentParser(description=desc)
    parser.add_argument('template', help='Template source/sink xml file.')
    parser.add_argument('output', nargs='?', help='Output xml file path.')
    parser.add_argument(
        '-n', '--max_channels', type=int, default=10,
        help='''Create block with the given maximum number of channels.
                (default: %(default)s)''',
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.basename(os.path.splitext(args.template)[0])

    tmpl = Template(filename=args.template)
    with open(args.output, 'w') as f:
        f.write(tmpl.render(max_num_channels=args.max_channels))
