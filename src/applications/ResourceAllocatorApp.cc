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

Define_Module(ResourceAllocatorApp);

void ResourceAllocatorApp::initialize(int stage)
{
    ApplicationBase::initialize(stage);
    // Without binding sockets, this app will only return a ICMP error that
    // indicates the incoming packet cannot reach the required socket - as the
    // app is yet to be on a socket.

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
    if (msg->getArrivalGate()->isName("socketIn")) {
        if (cPacket *pkt = dynamic_cast<cPacket *>(msg)) {
            EV << "Packet received: " << pkt->getName() << ", size: " << pkt->getByteLength() << " bytes" << endl;
            socket.processMessage(msg); // <- the one line of code that fixed it.
            delete pkt;
        } else {
            delete msg;
        }
    }

}

/**
 * =========================================================================
 * From UdpSocket::ICallback
 */
void ResourceAllocatorApp::socketDataArrived(UdpSocket *socket, Packet *packet)
{
    EV_INFO << "Received packet front chunk type: " << packet->peekAtFront()->getChunkType() << endl;

    auto taskRequirements =  packet->peekAtFront<MyTaskChunk>();
    auto cpuCycles = taskRequirements->getRequiredCPUCycles();
    auto deadlineLatency = taskRequirements->getDeadlineLatency();

    EV << "Payload All:" << packet->peekData() << endl;
    EV << "Data, CPU: " << cpuCycles <<
            ", Latency: " << deadlineLatency << endl;

    numReceived++;

    EV << "That was Packet " << numReceived << endl;
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
