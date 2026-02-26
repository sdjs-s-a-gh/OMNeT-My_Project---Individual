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

using namespace omnetpp;
using namespace inet;


/**
 * TODO - Generated class
 */
class ResourceAllocatorApp : public ApplicationBase, UdpSocket::ICallback
{
  protected:
    // Parameter
    int localPort = -1;

    // Statistics
    int numReceived = 0;

    UdpSocket socket; // Requires a socket to bind the application to.

    virtual void initialize(int stage) override;
    virtual void handleMessageWhenUp(cMessage *msg) override;

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
