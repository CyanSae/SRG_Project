# BSD 3-Clause License
#
# Copyright (c) 2016, 2017, The University of Sydney. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""exporter.py: abstract classes for exporting decompiler state"""

import abc
import csv
import logging
import os

import src.cfg as cfg
import src.function as function
import src.opcodes as opcodes
import src.patterns as patterns
import src.tac_cfg as tac_cfg


class Exporter(abc.ABC):
    def __init__(self, source: object):
        """
        Args:
          source: object instance to be exported
        """
        self.source = source

    @abc.abstractmethod
    def export(self):
        """
        Exports the source object to an implementation-specific format.
        """


class CFGTsvExporter(Exporter, patterns.DynamicVisitor):
    """
    Writes logical relations of the given TAC CFG to local directory.

    Args:
      cfg: the graph to be written to logical relations.
    """

    def __init__(self, cfg: tac_cfg.TACGraph):
        """
        Generates .facts files of the given TAC CFG to local directory.

        Args:
          cfg: source TAC CFG to be exported to separate fact files.
        """
        super().__init__(cfg)

        self.defined = []
        """
        A list of pairs (op.pc, variable) that specify variable definition sites.
        """

        self.reads = []
        """
        A list of pairs (op.pc, variable) that specify all usage sites.
        """

        self.writes = []
        """
        A list of pairs (op.pc, variable) that specify all write locations.
        """

        self.__output_dir = None

    def __generate(self, filename, entries):
        path = os.path.join(self.__output_dir, filename)
        with open(path, 'w') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            for e in entries:
                writer.writerow(e)

    def __generate_blocks_ops(self, out_opcodes):
        # Write a mapping from operation addresses to corresponding opcode names;
        # a mapping from operation addresses to the block they inhabit;
        # any specified opcode listings.
        ops = []
        block_nums = []
        op_rels = {opcode: list() for opcode in out_opcodes}

        for block in self.source.blocks:
            for op in block.tac_ops:
                ops.append((hex(op.pc), op.opcode.name))
                block_nums.append((hex(op.pc), block.ident()))
                if op.opcode.name in out_opcodes:
                    output_tuple = tuple([hex(op.pc)] +
                                         [arg.value.name for arg in op.args])
                    op_rels[op.opcode.name].append(output_tuple)

        self.__generate("op.facts", ops)
        self.__generate("block.facts", block_nums)

        for opcode in op_rels:
            self.__generate("op_{}.facts".format(opcode), op_rels[opcode])

    def __generate_edges(self):
        # Write out the collection of edges between instructions (not basic blocks).
        edges = [(hex(h.pc), hex(t.pc))
                 for h, t in self.source.op_edge_list()]
        self.__generate("edge.facts", edges)

    def __generate_entry_exit(self):
        # Entry points
        entry_ops = [(hex(b.tac_ops[0].pc),)
                     for b in self.source.blocks if len(b.preds) == 0]
        self.__generate("entry.facts", entry_ops)

        # Exit points
        exit_points = [(hex(op.pc),) for op in self.source.terminal_ops]
        self.__generate("exit.facts", exit_points)

    def __generate_def_use_value(self):
        # Mapping from variable names to the addresses they were defined at.
        define = []
        # Mapping from variable names to the addresses they were used at.
        use = []
        # Mapping from variable names to their possible values.
        value = []
        for block in self.source.blocks:
            for op in block.tac_ops:
                # If it's an assignment op, we have a def site
                if isinstance(op, tac_cfg.TACAssignOp):
                    define.append((op.lhs.name, hex(op.pc)))

                    # And we can also find its values here.
                    if op.lhs.values.is_finite:
                        for val in op.lhs.values:
                            value.append((op.lhs.name, hex(val)))

                if op.opcode != opcodes.CONST:
                    # The args constitute use sites.
                    for i, arg in enumerate(op.args):
                        name = arg.value.name
                        if not arg.value.def_sites.is_const:
                            # Argument is a stack variable, and therefore needs to be
                            # prepended with the block id.
                            name = block.ident() + ":" + name
                        # relation format: use(Var, PC, ArgIndex)
                        use.append((name, hex(op.pc), i+1))

            # Finally, note where each stack variable might have been defined,
            # and what values it can take on.
            # This includes some duplication for stack variables with multiple def
            # sites. This can be done marginally more efficiently.
            for var in block.entry_stack:
                if not var.def_sites.is_const and var.def_sites.is_finite:
                    name = block.ident() + ":" + var.name
                    for loc in var.def_sites:
                        define.append((name, hex(loc.pc)))

                    if var.values.is_finite:
                        for val in var.values:
                            value.append((name, hex(val)))

        self.__generate("def.facts", define)
        self.__generate("use.facts", use)
        self.__generate("value.facts", value)

    def __generate_function(self):
        # Mapping from blocks to the solidity function they're in (if any)
        in_function = []
        # A function id appears in this relation if it's private.
        private_function = []
        public_function_sigs = []

        f_e = self.source.function_extractor
        for i, f in enumerate(f_e.functions):
            for b in f.body:
                in_function.append((b.ident(), i))
            if f.is_private:
                private_function.append((i,))
            else:
                public_function_sigs.append((i, f.signature))

        self.__generate("in_function.facts", in_function)
        self.__generate("private_function.facts", private_function)
        self.__generate("public_function_sigs.facts", public_function_sigs)

    def __generate_dominators(self):
        pairs = sorted([(k, i) for k, v
                        in self.source.dominators(op_edges=False).items()
                        for i in v])
        self.__generate("dom1.facts", pairs)

        pairs = sorted(self.source.immediate_dominators(op_edges=True).items())
        self.__generate("imdom.facts", pairs)

        pairs = sorted([(k, i) for k, v
                        in self.source.dominators(post=True,
                                                  op_edges=True).items()
                        for i in v])
        self.__generate("pdom.facts", pairs)

        pairs = sorted(self.source.immediate_dominators(post=True,
                                                        op_edges=True).items())
        self.__generate("impdom.facts", pairs)

    def __generate_global_order(self):
        # Introduce a total order for IR instructions, allowing inductive
        # reasoning in Souffle. Simply number each IR instruction from 0 to n,
        # and use this global counter for relations BasicBlockRange, CFGEdge,
        # and the mapping from hex PCs to global counters: op_globalcount.
        # Relations:
        #   BasicBlockRange(block_ident, global_start, global_end)
        #   op_globalcount(op_hex, op_global_count)
        #   CFGEdge(global_block_entry, global_block_exit)
        counter = 0
        ops = []
        block_ranges = []
        entry, exit = None, None
        block_id_to_global_entry = {}
        for block in self.source.blocks:
            for i, op in enumerate(block.tac_ops):
                if i == 0:
                    entry = counter
                    block_id_to_global_entry[block.entry] = counter
                if i == len(block.tac_ops)-1:
                    exit = counter
                ops.append((hex(op.pc), counter))
                counter += 1
            block_ranges.append((hex(block.entry), entry, exit))

        cfg_edges = []
        for (u, v) in self.source.edge_list():
            cfg_edges.append((block_id_to_global_entry[u.entry],
                block_id_to_global_entry[v.entry]))

        self.__generate("op_globalcount.facts", ops)
        self.__generate("BasicBlockRange.facts", block_ranges)
        self.__generate("CFGEdge.facts", cfg_edges)

    def export(self, output_dir: str = "", dominators: bool = False, out_opcodes=[]):
        """
        Args:
          output_dir: location to write the output to.
          dominators: output relations specifying dominators
          out_opcodes: a list of opcode names all occurences thereof to output,
                       with the names of all argument variables.
        """
        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)
        self.__output_dir = output_dir

        self.__generate_blocks_ops(out_opcodes)
        self.__generate_edges()
        self.__generate_entry_exit()
        self.__generate_def_use_value()
        self.__generate_global_order()

        if self.source.function_extractor is not None:
            self.__generate_function()

        if dominators:
            self.__generate_dominators()


class CFGStringExporter(Exporter, patterns.DynamicVisitor):
    """
    Prints a textual representation of the given CFG to stdout.

    Args:
      cfg: source CFG to be printed.
      ordered: if True (default), print BasicBlocks in order of entry.
    """

    __BLOCK_SEP = "\n\n================================\n\n"

    def __init__(self, cfg: cfg.ControlFlowGraph, ordered: bool = True):
        super().__init__(cfg)
        self.ordered = ordered
        self.blocks = []
        self.source.accept(self)

    def visit_ControlFlowGraph(self, cfg):
        """
        Visit the CFG root
        """
        pass

    def visit_BasicBlock(self, block):
        """
        Visit a BasicBlock in the CFG
        """
        self.blocks.append((block.entry, str(block)))

    def export(self):
        """
        Print a textual representation of the input CFG to stdout.
        """
        if self.ordered:
            self.blocks.sort(key=lambda n: n[0])
        blocks = self.__BLOCK_SEP.join(n[1] for n in self.blocks)
        functions = ""
        if self.source.function_extractor is not None:
            functions = self.__BLOCK_SEP + str(self.source.function_extractor)
        return blocks + functions


class CFGDotExporter(Exporter):
    """
    Generates a dot file for drawing a pretty picture of the given CFG.

    Args:
      cfg: source CFG to be exported to dot format.
    """

    def __init__(self, cfg: cfg.ControlFlowGraph):
        super().__init__(cfg)

    def export(self, out_filename: str = "cfg.dot"):
        """
        Export the CFG to a dot file.

        Certain blocks will have coloured outlines:
          Green: contains a RETURN operation;
          Blue: contains a STOP operation;
          Red: contains a THROW, THROWI, INVALID, or missing operation;
          Purple: contains a SELFDESTRUCT operation;
          Orange: contains a CALL, CALLCODE, or DELEGATECALL operation;
          Brown: contains a CREATE operation.

        A node with a red fill indicates that its stack size is large.

        Args:
          out_filename: path to the file where dot output should be written.
                        If the file extension is a supported image format,
                        attempt to generate an image using the `dot` program,
                        if it is in the user's `$PATH`.
        """
        import networkx as nx

        cfg = self.source

        G = cfg.nx_graph()

        # Colour-code the graph.
        returns = {block.ident(): "green" for block in cfg.blocks
                   if block.last_op.opcode == opcodes.RETURN}
        stops = {block.ident(): "blue" for block in cfg.blocks
                 if block.last_op.opcode == opcodes.STOP}
        throws = {block.ident(): "red" for block in cfg.blocks
                  if block.last_op.opcode.is_exception()}
        suicides = {block.ident(): "purple" for block in cfg.blocks
                    if block.last_op.opcode == opcodes.SELFDESTRUCT}
        creates = {block.ident(): "brown" for block in cfg.blocks
                   if any(op.opcode == opcodes.CREATE for op in block.tac_ops)}
        calls = {block.ident(): "orange" for block in cfg.blocks
                 if any(op.opcode.is_call() for op in block.tac_ops)}
        color_dict = {**returns, **stops, **throws, **suicides, **creates, **calls}
        nx.set_node_attributes(G, "color", color_dict)
        filldict = {b.ident(): "white" if len(b.entry_stack) <= 20 else "red"
                    for b in cfg.blocks}
        nx.set_node_attributes(G, "fillcolor", filldict)
        nx.set_node_attributes(G, "style", "filled")

        # Annotate each node with its basic block's internal data for later display
        # if rendered in html.
        nx.set_node_attributes(G, "id", {block.ident(): block.ident()
                                         for block in cfg.blocks})

        block_strings = {}
        for block in cfg.blocks:
            block_string = str(block)
            def_site_string = "\n\nDef sites:\n"
            for v in block.entry_stack.value:
                def_site_string += str(v) \
                                   + ": {" \
                                   + ", ".join(str(d) for d in v.def_sites) \
                                   + "}\n"
            block_strings[block.ident()] = block_string + def_site_string
        nx.set_node_attributes(G, "tooltip", block_strings)

        # Write non-dot files using pydot and Graphviz
        if "." in out_filename and not out_filename.endswith(".dot"):
            pdG = nx.nx_pydot.to_pydot(G)
            extension = out_filename.split(".")[-1]

            # If we're producing an html file, write a temporary svg to build it from
            # and then delete it.
            if extension == "html":
                html = svg_to_html(pdG.create_svg().decode("utf-8"), cfg.function_extractor)
                if not out_filename.endswith(".html"):
                    out_filename += ".html"
                with open(out_filename, 'w') as page:
                    logging.info("Drawing CFG image to '%s'.", out_filename)
                    page.write(html)
            else:
                pdG.set_margin(0)
                pdG.write(out_filename, format=extension)

        # Otherwise, write a regular dot file using pydot
        else:
            try:
                if out_filename == "":
                    out_filename = "cfg.html"
                nx.nx_pydot.write_dot(G, out_filename)
                logging.info("Drawing CFG image to '%s'.", out_filename)
            except:
                logging.info("Graphviz missing. Falling back to dot.")
                if out_filename == "":
                    out_filename = "cfg.dot"
                nx.nx_pydot.write_dot(G, out_filename)
                logging.info("Drawing CFG image to '%s'.", out_filename)


def svg_to_html(svg: str, function_extractor: function.FunctionExtractor = None) -> str:
    """
    Produces an interactive html page from an svg image of a CFG.

    Args:
        svg: the string of the SVG to process
        function_extractor: a FunctionExtractor object containing functions
                            to annotate the graph with.

    Returns:
        HTML string of interactive web page source for the given CFG.
    """

    lines = svg.split("\n")
    page = []

    page.append("""
              <html>
              <body>
              <style>
              .node
              {
                transition: all 0.05s ease-out;
              }
              .node:hover
              {
                stroke-width: 1.5;
                cursor:pointer
              }
              .node:hover
              ellipse
              {
                fill: #EEE;
              }
              textarea#infobox {
                position: fixed;
                display: block;
                top: 0;
                right: 0;
              }

              .dropbutton {
                padding: 10px;
                border: none;
              }
              .dropbutton:hover, .dropbutton:focus {
                background-color: #777777;
              }
              .dropdown {
                margin-right: 5px;
                position: fixed;
                top: 5px;
                right: 0px;
              }
              .dropdown-content {
                background-color: white;
                display: none;
                position: absolute;
                width: 70px;
                box-shadow: 0px 5px 10px 0px rgba(0,0,0,0.2);
                z-index: 1;
              }
              .dropdown-content a {
                color: black;
                padding: 8px 10px;
                text-decoration: none;
                font-size: 10px;
                display: block;
              }

              .dropdown-content a:hover { background-color: #f1f1f1; }

              .show { display:block; }
              </style>
              """)

    for line in lines[3:]:
        page.append(line)

    page.append("""<textarea id="infobox" disabled=true rows=40 cols=80></textarea>""")

    # Create a dropdown list of functions if there are any.
    if function_extractor is not None:
        page.append("""<div class="dropdown">
               <button onclick="showDropdown()" class="dropbutton">Functions</button>
               <div id="func-list" class="dropdown-content">""")

        for i, f in enumerate(function_extractor.functions):
            if f.is_private:
                page.append('<a id=f_{0} href="javascript:highlightFunction({0})">private #{0}</a>'.format(i))
            else:
                if f.signature:
                    page.append(
                        '<a id=f_{0} href="javascript:highlightFunction({0})">public {1}</a>'.format(i, f.signature))
                else:
                    page.append('<a id=f_{0} href="javascript:highlightFunction({0})">fallback</a>'.format(i))
        page.append("</div></div>")

    page.append("""<script>""")

    if function_extractor is not None:
        func_map = {i: [b.ident() for b in f.body]
                    for i, f in enumerate(function_extractor.functions)}
        page.append("var func_map = {};".format(func_map))
        page.append("var highlight = new Array({}).fill(0);".format(len(func_map)))

    page.append("""
               // Set info textbox contents to the title of the given element, with line endings replaced suitably.
               function setInfoContents(element){
                   document.getElementById('infobox').value = element.getAttribute('xlink:title').replace(/\\\\n/g, '\\n');
               }

               // Make all node anchor tags in the svg clickable.
               for (var el of Array.from(document.querySelectorAll(".node a"))) {
                   el.setAttribute("onclick", "setInfoContents(this);");
               }

               const svg = document.querySelector('svg')
               const NS = "http://www.w3.org/2000/svg";
               const defs = document.createElementNS( NS, "defs" );

               // IIFE add filter to svg to allow shadows to be added to nodes within it
               (function(){
                 defs.innerHTML = makeShadowFilter()
                 svg.insertBefore(defs,svg.children[0])
               })()

               function colorToID(color){
                 return color.replace(/[^a-zA-Z0-9]/g,'_')
               }

               function makeShadowFilter({color = 'black',x = 0,y = 0, blur = 3} = {}){
                 return `
                 <filter id="filter_${colorToID(color)}" x="-40%" y="-40%" width="250%" height="250%">
                   <feGaussianBlur in="SourceAlpha" stdDeviation="${blur}"/>
                   <feOffset dx="${x}" dy="${y}" result="offsetblur"/>
                   <feFlood flood-color="${color}"/>
                   <feComposite in2="offsetblur" operator="in"/>
                   <feMerge>
                     <feMergeNode/>
                     <feMergeNode in="SourceGraphic"/>
                   </feMerge>
                 </filter>
                 `
               }

               // Shadow toggle functions, with filter caching
               function addShadow(el, {color = 'black', x = 0, y = 0, blur = 3}){
                 const id = colorToID(color);
                 if(!defs.querySelector(`#filter_${id}`)){
                   const d = document.createElementNS(NS, 'div');
                   d.innerHTML = makeShadowFilter({color, x, y, blur});
                   defs.appendChild(d.children[0]);
                 }
                 el.style.filter = `url(#filter_${id})`
               }

               function removeShadow(el){
                 el.style.filter = ''
               }

               function hash(n) {
                 var str = n + "rainbows" + n + "please" + n;
                 var hash = 0;
                 for (var i = 0; i < str.length; i++) {
                   hash = (((hash << 5) - hash) + str.charCodeAt(i)) | 0;
                 }
                 return hash > 0 ? hash : -hash;
               };

               function getColor(n, sat="80%", light="50%") {
                 const hue = hash(n) % 360;
                 return `hsl(${hue}, ${sat}, ${light})`;
               }

               // Add shadows to function body nodes, and highlight functions in the dropdown list
               function highlightFunction(i) {
                 for (var n of Array.from(document.querySelectorAll(".node ellipse"))) {
                   removeShadow(n);
                 }

                 highlight[i] = !highlight[i];
                 const entry = document.querySelector(`.dropdown-content a[id='f_${i}']`)
                 if (entry.style.backgroundColor) {
                   entry.style.backgroundColor = null;
                 } else {
                   entry.style.backgroundColor = getColor(i, "60%", "90%");
                 }

                 for (var j = 0; j < highlight.length; j++) {
                   if (highlight[j]) {
                     const col = getColor(j);
                     for (var id of func_map[j]) {
                       var n = document.querySelector(`.node[id='${id}'] ellipse`);
                       addShadow(n, {color:`${col}`});
                     }
                   }
                 }
               }

               // Show the dropdown elements when it's clicked.
               function showDropdown() {
                 document.getElementById("func-list").classList.toggle("show");
               }
               window.onclick = function(event) {
                 if (!event.target.matches('.dropbutton')) {
                   var items = Array.from(document.getElementsByClassName("dropdown-content"));
                   for (var item of items) {
                     item.classList.remove('show');
                   }
                 }
               }
              </script>
              </html>
              </body>
              """)

    return "\n".join(page)
