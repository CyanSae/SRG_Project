This directory contains examples of raw `disasm` output which may be used as input to the decompiler.

python bin/decompile -t results/owner/tsv -d examples/owner.hex results/owner/decompile.txt

```shell
bin/decompile -g results/inverse_finance/graph.html examples/instances/2_inverse_finance.hex results/inverse_finance/decompile.txt
bin/disassemble examples/instances/2_inverse_finance.hex -o results/inverse_finance/disassemble.txt
bin/decompile -t results/owner/tsv -d examples/owner.hex results/owner/decompile.txt
python bin/decompile -t results/basic/tsv -d examples/basic.hex results/basic/decompile.txt
python bin/decompile -g results/dao_hack/graph.html -t results/dao_hack/tsv -d examples/dao_hack.hex results/dao_hack/decompile.txt
python SOG/bin/decompile -g SOG/results/322.html -t SOG/results/322/tsv -d SOG/examples/322.hex SOG/results/322/decompile.txt
python SOG/bin/decompile -d SOG/examples/322.hex SOG/results/322/decompile.txt
```


2. Inverse Finance
https://etherscan.io/address/0xf508c58ce37ce40a40997C715075172691F92e2D#code
![alt text](fee28000a7e631e780fdd82f11d4e9a3.png)

3. MEV Bot: 0xE9...0d6
https://etherscan.io/address/0xE911519dc7f35996C6ad5C8A53e82B101af790d6#code

4. C2 Server in Smargaft
https://bscscan.com/address/0xdf2208d4902aa1ec9a0957132ca86a4e1d40455b#code
