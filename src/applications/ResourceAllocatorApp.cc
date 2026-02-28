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

#include "ResourceAllocatorApp.h"
#include "MyTaskChunk_m.h"
//#include "EndTaskExecution_m.h"

Define_Module(ResourceAllocatorApp);

void ResourceAllocatorApp::initialize(int stage)
{
    ApplicationBase::initialize(stage);
    // Without binding sockets, this app will only return a ICMP error that
    // indicates the incoming packet cannot reach the required socket - as the
    // app is yet to be on a socket.


    maxCPUCapacity = par("maxCPUCapacity");
    currentCapacity = maxCPUCapacity;

    // Setup Statistics
    lifeTimeSignal = registerSignal("lifeTime");
    totalQueueingTimeSignal = registerSignal("totalQueueingTime");
    queuesVisitedSignal = registerSignal("queuesVisited");
    totalServiceTimeSignal = registerSignal("totalServiceTime");
    totalDelayTimeSignal = registerSignal("totalDelayTime");
    delaysVisitedSignal = registerSignal("delaysVisited");

    if (stage == INITSTAGE_APPLICATION_LAYER) {
        localPort = par("localPort");
        socket.setOutputGate(gate("socketOut"));
        socket.bind(localPort);
        socket.setCallback(this);
        EV << "EdgeServerResourceAllocatorApp has been successfully set up on Port: " << localPort << endl;
    }


}

void ResourceAllocatorApp::handleMessageWhenUp(cMessage *msg)
{
//    if (cPacket *pkt = dynamic_cast<cPacket *>(msg)) {
//        EV << "Packet received: " << pkt->getName() << ", size: " << pkt->getByteLength() << " bytes" << endl;
//        socket.processMessage(msg); // <- the one line of code that fixed it.
//
////            emit(lifeTimeSignal, simTime() - job->getCreationTime());
////            emit(totalQueueingTimeSignal, job->getTotalQueueingTime());
////            emit(queuesVisitedSignal, job->getQueueCount());
////            emit(totalServiceTimeSignal, job->getTotalServiceTime());
////            emit(totalDelayTimeSignal, job->getTotalDelayTime());
////            emit(delaysVisitedSignal, job->getDelayCount());
//            //delete pkt;
//    } else {
//            //delete msg;
//    }


    // If a task has completed.
    if (msg->isSelfMessage() && strcmp(msg->getName(), "endExecutionSelfMessage") == 0) {
        endTaskExecution(msg);
    // Incoming Task
    } else if (msg->arrivedOn("socketIn")) {
        // Process a Packet
        socket.processMessage(msg);
    } else {
        throw cRuntimeError("Unknown message received on gate: %s", msg->getArrivalGate()->getFullName());
    }

    // Regardless of the message type (not providing an erroneous type), update the queue
    // to check whether the next task can be processed.
    updateQueue();
    delete msg;
}

/**
 * Allocates CPU cycles to the task passed as an argument.
 */
void ResourceAllocatorApp::allocateResources(Task *task)
{
    int cpuCyclesToAllocate = task->requiredCPUCycles;
    task->allocatedCPUCycles = cpuCyclesToAllocate;
    task->executionTime = getTimeToExecute(cpuCyclesToAllocate);

    task->startServiceTime = simTime();
}

/**
 * Returns the amount of time in seconds it will take to
 * execute a task of a certain size based on the capacity of
 * the edge server.
 */
double ResourceAllocatorApp::getTimeToExecute(double cpuCycles)
{
    return cpuCycles / maxCPUCapacity;
}

/**
 * The subroutine that processes the input task.
 *
 * The subroutine takes the task to be executed on the CPU
 * and subsequently deallocates the CPU cycles required from
 * the current capacity of the edge server.
 */
void ResourceAllocatorApp::processTask(Task *task)
{
    EV << "========" << endl << "processTask" << endl;
    auto *endExecutionSelfMessage = new cMessage("endExecutionSelfMessage");
    endExecutionSelfMessage->setContextPointer(task);

    scheduleAt(simTime() + task->executionTime, endExecutionSelfMessage);
    currentCapacity -= task->allocatedCPUCycles;
    simtime_t scheduleTimeTotal = simTime() + task->executionTime;
    simtime_t scheduleTimeInMS = scheduleTimeTotal * 1000;

    EV << "Processing Task. Current Capacity: " << currentCapacity << endl;
    EV << "That will take " << task->executionTime << " seconds. According to the simulator, that will be at "
            << SIMTIME_STR(scheduleTimeTotal) << ", which is at " << scheduleTimeInMS << "ms." << endl;
    tasksProcessing++;
    EV << "Tasks Currently Processing: " << tasksProcessing << endl;
}

/**
 * A subroutine that finishes the execution of a task, freeing-up
 * CPU space.
 */
void ResourceAllocatorApp::endTaskExecution(cMessage *msg)
{
    EV << "========" << endl << "endTaskExecution" << endl;
    EV << "Received Self-Message. Finishing task. ";

    auto *completedTask = static_cast<Task *>(msg->getContextPointer());

    completedTask->endServiceTime = simTime();
    completedTask->totalServiceTime = completedTask->endServiceTime - completedTask->startServiceTime;
    // emit(msg->getTotalServiceTime);
    // totalTime? = communication delay + queue time + computation time?

    // Free up the CPU of the edge server by giving back the allocated CPU cycles.
    currentCapacity += completedTask->allocatedCPUCycles;
    tasksProcessing--;
    tasksProcessed++;

    EV << "Capacity is now: " << currentCapacity << endl;
    EV << "Began Servicing: " << SIMTIME_STR(completedTask->startServiceTime) << "; Finished Servicing: " << SIMTIME_STR(completedTask->endServiceTime) << endl;
    EV << "The total service time for that task was: " << completedTask->totalServiceTime << "seconds or " << completedTask->totalServiceTime * 1000 << "ms." << endl;
    EV << "Tasks Processed: " << tasksProcessed << endl;

    //delete completedTask;
}

/**
 * A subroutine that checks whether the next task in the queue can be processed.
 */
void ResourceAllocatorApp::updateQueue()
{
    EV << "========" << endl << "updateQueue" << endl;

    if (!queue.isEmpty()) {
        auto *queueHead = check_and_cast<Task *>(queue.front());

        EV << "Is the Queue Empty? " << queue.isEmpty() << endl;
        EV << "Current Capacity: " << currentCapacity << endl;

        // Keep processing tasks while there is enough resources remaining.
        while (!queue.isEmpty() && queueHead->allocatedCPUCycles <= currentCapacity) {
            processTask(queueHead);

            // Only now remove (dequeue) the task from the queue once there are enough resources.
            queue.pop();

            // Update the head of the queue (Don't think I need to as it would have been done by dequeueing)
            // queueHead.front();
            }
        EV << "The length of the queue is: " << queue.getLength() << endl;;
    }

}


/**
 * =========================================================================
 * From UdpSocket::ICallback
 */
void ResourceAllocatorApp::socketDataArrived(UdpSocket *socket, Packet *packet)
{
    EV << "========"<< endl << "socketDataArrived" << endl;
    EV << "Packet received: " << packet->getName() << ", size: " << packet->getByteLength() << " bytes" << endl;

    auto taskRequirements =  packet->peekAtFront<MyTaskChunk>();
    auto cpuCycles = taskRequirements->getRequiredCPUCycles();
    auto deadlineLatency = taskRequirements->getDeadlineLatency();

    numReceived++;

    EV << "That was Packet " << numReceived << endl;

    // Retrieve the task requirements from the Packet's payload.
    auto taskChunk = packet->peekAtFront<MyTaskChunk>();

    // As Chunks are immutable, copy the data over from them.
    auto *task = new Task();
    task->arrivalTime = simTime();
    task->requiredCPUCycles = taskChunk->getRequiredCPUCycles();

    allocateResources(task);
    EV << "CPU Cycles Allocated: " << task->allocatedCPUCycles << endl;
    queue.insert(task); // enqueue the task
}

void ResourceAllocatorApp::refreshDisplay() const
{
    ApplicationBase::refreshDisplay();

    char buf[50];
    sprintf(buf, "Packets Received: %d", numReceived);
    getDisplayString().setTagArg("t", 0, buf);
}

void ResourceAllocatorApp::socketErrorArrived(UdpSocket *socket, Indication *indication)
{
    EV_WARN << "Ignoring UDP error report " << indication->getName() << endl;
    delete indication;
}

void ResourceAllocatorApp::socketClosed(UdpSocket *socket)
{

}

/**
 * ==========================================================================
 * From Application Base
 */
void ResourceAllocatorApp::handleStartOperation(LifecycleOperation *operation)
{

}

void ResourceAllocatorApp::handleStopOperation(LifecycleOperation *operation)
{

}

void ResourceAllocatorApp::handleCrashOperation(LifecycleOperation *operation)
{

}
