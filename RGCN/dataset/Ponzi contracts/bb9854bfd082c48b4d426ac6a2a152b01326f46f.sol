/**
 *Submitted for verification at Etherscan.io on 2016-03-26
*/
contract X3 {
        struct Participant {
                address etherAddress;
                uint amount;
        }
        Participant[] public participants;
        uint public payoutIdx = 0;
        uint public collectedFees;
        uint public balance = 0;
        address public owner;
        // simple single-sig function modifier
        modifier onlyowner {
                if (msg.sender == owner) _
        }
        // this function is executed at initialization and sets the owner of the contract
        function X3() {
                owner = msg.sender;
        }
        // fallback function - simple transactions trigger this
        function() {
                enter();
        }
        function enter() {
                if (msg.value < 1 ether) {
                        msg.sender.send(msg.value);
                        return;
                }
                // add a new participant to array
                uint idx = participants.length;
                participants.length += 1;
                participants[idx].etherAddress = msg.sender;
                participants[idx].amount = msg.value;
                // collect fees and update contract balance
                if (idx != 0) {
                        collectedFees += msg.value / 3;
                        balance += msg.value;
                } else {
                        // first participant has no one above him,
                        // so it goes all to fees
                        collectedFees += msg.value;
                }
                // if there are enough ether on the balance X3 will payout three time your initial investement
                if (balance > participants[payoutIdx].amount * 3) {
                        uint transactionAmount = 3 * (participants[payoutIdx].amount - participants[payoutIdx].amount / 3);
                        participants[payoutIdx].etherAddress.send(transactionAmount);
                        balance -= participants[payoutIdx].amount * 3;
                        payoutIdx += 1;
                }
        }
        function collectFees() onlyowner {
                if (collectedFees == 0) return;
                owner.send(collectedFees);
                collectedFees = 0;
        }
        function setOwner(address _owner) onlyowner {
                owner = _owner;
        }
}