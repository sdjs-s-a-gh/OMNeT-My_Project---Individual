//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

#ifndef __MYPROJECTANDTUTORIAL_RESOURCEALLOCATORAPP_H_
#define __MYPROJECTANDTUTORIAL_RESOURCEALLOCATORAPP_H_

#include <omnetpp.h>
#include <inet/applications/base/ApplicationBase.h>
#include "inet/transportlayer/contract/udp/UdpSocket.h"
#include "MyTaskChunk_m.h"

using namespace omnetpp;
using namespace inet;


/**
 * Better than a class since I need to access and modify everything.
 */
struct Task : public cObject
{
    double requiredCPUCycles;
    int deadlineLatency;
    double allocatedCPUCycles;
    double executionTime;


    // Variables Mainly for Statistics
    simtime_t arrivalTime;  // The time at which the task entered the edge server - just before it is allocated resources.
    simtime_t startServiceTime; // The time at which the task has just been allocated resources - before it is queued and then executed.
    simtime_t endServiceTime; // The time at which the task has finished executing on the edge server.
    simtime_t totalServiceTime; // The total duration of the task's queueing + execution/processing time.
};


/**
 * TODO - Generated class
 */
class ResourceAllocatorApp : public ApplicationBase, UdpSocket::ICallback
{
  protected:
    // Resource Allocation
    double maxCPUCapacity = 10000; // in MHz currently
    double currentCapacity = 0;
    cQueue queue;

    // Parameter
    int localPort = -1;

    // Statistics
    int numReceived = 0;
    int tasksProcessed = 0;
    int tasksProcessing = 0;

    simsignal_t lifeTimeSignal;
    simsignal_t totalQueueingTimeSignal;
    simsignal_t queuesVisitedSignal;
    simsignal_t totalServiceTimeSignal;
    simsignal_t totalDelayTimeSignal;
    simsignal_t delaysVisitedSignal;

    UdpSocket socket; // Requires a socket to bind the application to.

    void processTask(Task *task);
    double getTimeToExecute(double cpuCycles);
    void allocateResources(Task *task);
    void endTaskExecution(cMessage *msg);
    void updateQueue();

    virtual void initialize(int stage) override;
    virtual void handleMessageWhenUp(cMessage *msg) override;

    virtual void refreshDisplay() const override;

    // Inheriting UdpSocket::ICallback requires these 3 methods.
    virtual void socketDataArrived(UdpSocket *socket, Packet *packet) override;
    virtual void socketErrorArrived(UdpSocket *socket, Indication *indication) override;
    virtual void socketClosed(UdpSocket *socket) override;

    // Inheriting ApplicationBase requires these 3 methods.
    virtual void handleStartOperation(LifecycleOperation *operation) override;
    virtual void handleStopOperation(LifecycleOperation *operation) override;
    virtual void handleCrashOperation(LifecycleOperation *operation) override;

  public:
    virtual int numInitStages() const override { return NUM_INIT_STAGES; }
};

#endif
