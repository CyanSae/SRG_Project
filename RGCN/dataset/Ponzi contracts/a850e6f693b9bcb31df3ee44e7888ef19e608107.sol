/**
 *Submitted for verification at Etherscan.io on 2016-04-04
*/
contract plusOnePonzi {
  uint public constant VALUE = 9 ether;
  struct Payout {
    address addr;
    uint yield;
  }
  Payout[] public payouts;
  uint public payoutIndex = 0;
  uint public payoutTotal = 0;
  function PlusOnePonzi() {
  }
  function() {
    if (msg.value < VALUE) {
      throw;
    }
    uint entryIndex = payouts.length;
    payouts.length += 1;
    payouts[entryIndex].addr = msg.sender;
    payouts[entryIndex].yield = 10 ether;
    while (payouts[payoutIndex].yield < this.balance) {
      payoutTotal += payouts[payoutIndex].yield;
      payouts[payoutIndex].addr.send(payouts[payoutIndex].yield);
      payoutIndex += 1;
    }
  }
}