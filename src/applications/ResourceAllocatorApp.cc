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

    WATCH(currentCapacity);
    WATCH(packetsReceived);
    WATCH(tasksProcessed);
    WATCH(tasksProcessing);
    WATCH(maxQueueLength);

    // Setup Statistics
    latencySignal = registerSignal("latency");
    resourceUtilisationSignal = registerSignal("resourceUtilisation");
    energyConsumptionSignal = registerSignal("energyConsumption");

    tasksProcessedSignal = registerSignal("tasksProcessed");
    maxQueueLengthSignal = registerSignal("maxQueueLength");
    parallelTasksSignal = registerSignal("parallelTasks");

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
    // Would need to get resource utilisation here before allocating resources.
    int cpuCyclesToAllocate = task->requiredCPUCycles;
    task->allocatedCPUCycles = cpuCyclesToAllocate;
    task->executionTime = getTimeToExecute(cpuCyclesToAllocate);
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
 * Returns the current resource utilisation of the edge server - between 0 and 1.
 */
double ResourceAllocatorApp::getResourceUtilisation()
{
    return 1.0 - (currentCapacity / maxCPUCapacity);
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

    completedTask->completionTime = simTime();
    completedTask->totalServiceTime = completedTask->completionTime - completedTask->arrivalTime;
    completedTask->latency = completedTask->completionTime - completedTask->creationTime;
    double energyConsumption = completedTask->allocatedCPUCycles * (maxCPUCapacity * maxCPUCapacity);
    completedTask->energyConsumption = energyConsumption;
    // emit(msg->getTotalServiceTime);
    // totalTime? = communication delay + queue time + computation time?

    // Free up the CPU of the edge server by giving back the allocated CPU cycles.
    currentCapacity += completedTask->allocatedCPUCycles;
    tasksProcessing--;
    tasksProcessed++;

    EV << "Capacity is now: " << currentCapacity << endl;
    // These print statements could be inside a function of Task instead.
    EV << "Task was Created" << SIMTIME_STR(completedTask->creationTime) << "; Completion Time:" << SIMTIME_STR(completedTask->completionTime) << endl;
    EV << "Began Servicing: " << SIMTIME_STR(completedTask->arrivalTime) << "; Finished Servicing: " << SIMTIME_STR(completedTask->completionTime) << endl;
    EV << "The total service time for that task was: " << completedTask->totalServiceTime << "seconds." << endl;
    EV << "Tasks Processed: " << tasksProcessed << endl;
    EV << "Task Latency: " << completedTask->latency << " seconds, or " << completedTask->latency * 1000 << " ms." << endl;
    EV << "Task Energy Consumption: " << completedTask->energyConsumption << endl;

    double latency = completedTask->latency.dbl() * 1000; // convert to milliseconds.
    emit(latencySignal,latency);
    emit(energyConsumptionSignal, energyConsumption);
    emit(tasksProcessedSignal, tasksProcessed);
    emit(parallelTasksSignal, tasksProcessing);
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
            }
        EV << "The length of the queue is: " << queue.getLength() << endl;
        emit(maxQueueLengthSignal, maxQueueLength);
        EV << "Resource Utilisation: " << getResourceUtilisation() << endl;
        emit(resourceUtilisationSignal, getResourceUtilisation());
        if (queue.getLength() > maxQueueLength) {
            maxQueueLength = queue.getLength();
        }
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

    EV << "That was Packet " << packetsReceived << endl;

    // Retrieve the task requirements from the Packet's payload.
    auto taskChunk = packet->peekAtFront<MyTaskChunk>();

    // As Chunks are immutable, copy the data over from them into a new variable.
    auto *task = new Task();
    task->requiredCPUCycles = taskChunk->getRequiredCPUCycles();

    task->arrivalTime = simTime();
    task->communicationDelay = task->arrivalTime - taskChunk->getCreationTime();
    task->creationTime = taskChunk->getCreationTime();

    EV << "Communication Delay: " << task->communicationDelay << endl;

    allocateResources(task);
    EV << "CPU Cycles Allocated: " << task->allocatedCPUCycles << endl;
    queue.insert(task); // enqueue the task
    packetsReceived++;
}

void ResourceAllocatorApp::refreshDisplay() const
{
    ApplicationBase::refreshDisplay();

    char buf[50];
    sprintf(buf, "Packets Received: %d", packetsReceived);
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
