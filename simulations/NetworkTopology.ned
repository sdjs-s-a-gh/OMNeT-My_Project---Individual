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

package myprojectandtutorial.simulations;

// Local Imports
import myprojectandtutorial.devices.EdgeServer;
import myprojectandtutorial.devices.MobilePhone;
import myprojectandtutorial.devices.AutonomousVehicle;

// Wireless Connections
import inet.node.inet.WirelessHost;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.physicallayer.wireless.common.contract.packetlevel.IRadioMedium;
// Visualing the Traffic Propagate throughout the medium
import inet.visualizer.canvas.integrated.IntegratedCanvasVisualizer;

// Physical Connections
import inet.node.inet.StandardHost;
import inet.node.inet.Router;
import inet.node.ethernet.Eth10M;

//import inet.physicallayer.wireless.common.medium.RadioMedium; // Required for wireless

network ResourceNetwork
{
    parameters:
        @display("bgb=600,600");

        // Statistics to be gathered during the simulation.
        // Main
        @statistic[latencyOverTime](source=latency; record=vector);
        @statistic[systemResourceUtilisationOverTime](source=resourceUtilisation; record=vector);
        @statistic[energyConsumptionOverTime](source=energyConsumption; record=vector);
	
        // Complementary
        @statistic[tasksProcessedOverTime](source=tasksProcessed; record=vector);
        @statistic[maxQueueLengthOverTime](source=maxQueueLength; record=vector);
        @statistic[parallelTasksOverTime](source=parallelTasks; record=vector);

    submodules:
        visualizer: IntegratedCanvasVisualizer {
            @display("p=62,475");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=62,553");
        }
        radioMedium: <default("ApskScalarRadioMedium")> like IRadioMedium {
            @display("p=62,400");
        } // Will need to change to RadioMedium to configure bandwidth

        // User Layer
        hostA: WirelessHost {
            @display("p=50,195");
        }

        mdMobilePhone: MobilePhone {
            @display("p=230,291");
        }

        mdAutonomousVehicle[1]: AutonomousVehicle {
            @display("p=428,289");
        }

        //hostB: WirelessHost { @display("p=450,195"); }

        // Edge Layer
        baseStation: WirelessHost {
            @display("p=250,150");
        }
        coreRouter: Router {
            @display("p=250,50");
        }
        edgeServer: EdgeServer {
            @display("p=500,50");
        } // Was StandardHost

        // Cloud Layer
		// cloudServer: WirelessHost


    // Physical Connections
    connections:
        baseStation.ethg++ <--> Eth10M <--> coreRouter.ethg++;
        coreRouter.ethg++ <--> Eth10M <--> edgeServer.ethg++;

}