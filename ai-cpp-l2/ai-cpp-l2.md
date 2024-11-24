# Image crop and resize

In this lesson, we will learn how to crop and resize images using CPP.

## We will cover the following topics:
- Cache locality
- SIMD - execution_unsequenced_policy
- Implace operations
- Constexpr
- OpenCV


## Cache locality
One cycle on a 3 GHz proccessor - 1ns.
L1 cache - 32KB.
- Access to L1 cache - 0.5ns.
- Branch misprediction - 5ns.
L2 cache - 256KB. 
- Access to L2 cache - 7ns.
Mutex Lock/Unlock - 25ns.
Access to RAM - 100ns.
Access to SSD - 10^6ns.
Reading 1MB from RAM - 10^6ns.
Sending 1MB over the network - 10^6ns.
https://share.evernote.com/note/e903a4d9-9fa2-e33f-00df-84fd6ee0e18a