// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title RecordAnchor
/// @notice Simple contract to anchor SHA-256 record hashes on Ethereum Mainnet.
contract RecordAnchor {
    event HashStored(address indexed sender, bytes32 indexed recordHash, uint256 timestamp);

    mapping(bytes32 => bool) public stored;

    /// @notice Store a record hash on-chain for later verification.
    /// @param recordHash A 32-byte hash, for example SHA-256 hash of a record.
    function storeHash(bytes32 recordHash) external {
        require(recordHash != bytes32(0), "Invalid hash");
        stored[recordHash] = true;
        emit HashStored(msg.sender, recordHash, block.timestamp);
    }

    /// @notice Verify whether a hash has been anchored.
    /// @param recordHash The 32-byte hash to check.
    /// @return anchored True if the hash was stored previously.
    function verifyHash(bytes32 recordHash) external view returns (bool anchored) {
        return stored[recordHash];
    }
}
