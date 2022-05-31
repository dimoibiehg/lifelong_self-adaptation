package smc.runmodes;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;

import mapek.AdaptationOption;
import mapek.Environment;
import mapek.Link;
import mapek.Mote;
import mapek.SNR;
import mapek.TrafficProbability;
import util.ConfigLoader;


public class ActivForms extends SMCConnector {

    public ActivForms() {
    }

    // @Override
    // public void startVerification() {
    //     initializeTimer();

    //     int verifiedOptions = 0;

    //     long startTime = System.currentTimeMillis();
	// 	for (AdaptationOption adaptationOption : adaptationOptions) {
    //         if (overTime) {
    //             break;
    //         }

    //         smcChecker.checkCAO(
	// 	            adaptationOption.toModelString(),
    //                 environment.toModelString(),
    //                 adaptationOption.verificationResults
    //         );
	// 		adaptationOption.isVerified = true;
	// 		verifiedOptions++;
	// 	}
	// 	long endTime = System.currentTimeMillis() - startTime;

    //     System.out.print(";" + verifiedOptions + ";" + endTime);

    //     destructTimer();
    // }

    @Override
    public void startVerification() {
		// Only collect data every 20 cycles
		// if (cycles % 20 != 0) {
		// 	for (AdaptationOption ao : adaptationOptions) {
		// 		ao.isVerified = true;
		// 		ao.verificationResults.packetLoss = 5;
		// 		ao.verificationResults.latency = 2;
		// 		ao.verificationResults.energyConsumption = 20;
		// 	}
		// 	return;
		// }

		Long[] verifTimes = new Long[adaptationOptions.size()]; 
		int index = 0;

		// Check all the adaptation options with activFORMS (and keep track of the verification time of each option)
		for (AdaptationOption adaptationOption : adaptationOptions) {
			Long startTime = System.currentTimeMillis();

			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
				adaptationOption.verificationResults);
			adaptationOption.isVerified = true;

			verifTimes[index] = System.currentTimeMillis() - startTime;
			index++;
		}


		storeAllFeaturesAndTargets(adaptationOptions, environment, cycles, verifTimes);
    }

    private void storeAllFeaturesAndTargets(List<AdaptationOption> adaptationOptions, Environment env, int cycle, Long[] verifTimes) {
        // Store the features and the targets in their respective files
        File dataset_file = new File(Paths.get(
                "./output/",
                ConfigLoader.getInstance().getSimulationNetwork().toLowerCase(), 
                "dataset_with_all_features" + cycle + ".json").toString());

        if (dataset_file.exists()) {
            // At the first cycle, remove the file if it already exists
            dataset_file.delete();
        }

        try {
            dataset_file.createNewFile();
            JSONObject root = new JSONObject();
            root.put("verification_times", new JSONArray());
            root.put("features", new JSONArray());
            // root.put("target_classification_packetloss", new JSONArray());
            root.put("packetloss", new JSONArray());
            // root.put("target_classification_latency", new JSONArray());
            root.put("latency", new JSONArray());
            root.put("energyconsumption", new JSONArray());
            FileWriter writer = new FileWriter(dataset_file);
            writer.write(root.toString(2));
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException(
                    String.format("Could not create the output file at %s", dataset_file.toPath().toString()));
        }

        try {
            JSONTokener tokener = new JSONTokener(dataset_file.toURI().toURL().openStream());
            JSONObject root = new JSONObject(tokener);

            // Get all the features for all the adaptation options, as well as their targets
            for (AdaptationOption option : adaptationOptions) {
                JSONArray newFeatures = new JSONArray();

                // Collection of all the features: SNR - Power setting - Distribution - Traffic probability
                for (SNR snr : env.linksSNR) {
                    newFeatures.put(snr.SNR);
                }

                option.system.motes.values().stream()
                        .map(mote -> mote.getLinks())
                        .flatMap(links -> links.stream())
                        .forEach(link -> newFeatures.put(link.getPower()));

                for (Mote mote : option.system.motes.values()) {
                    for (Link link : mote.getLinks()) {
                        newFeatures.put(link.getDistribution());
                    }
                }

                for (TrafficProbability traffic : env.motesLoad) {
                    newFeatures.put(traffic.load);
                }

                // Features
                root.getJSONArray("features").put(newFeatures);

                // TODO just include raw quality values instead of evaluating the goals as well

                // Packet loss values
                // root.getJSONArray("target_classification_packetloss").put(
                //         goals.getPacketLossGoal().evaluate(option.verificationResults.packetLoss) ? 1 : 0);
                root.getJSONArray("packetloss").put(option.verificationResults.packetLoss);

                // Latency values
                // root.getJSONArray("target_classification_latency").put(
                //         goals.getLatencyGoal().evaluate(option.verificationResults.latency) ? 1 : 0);
                root.getJSONArray("latency").put(option.verificationResults.latency);

                // Energy consumption values
                root.getJSONArray("energyconsumption").put(option.verificationResults.energyConsumption);
            }

            for (Long verifTime : verifTimes) {
                root.getJSONArray("verification_times").put(verifTime);
            }

            FileWriter writer = new FileWriter(dataset_file);
            writer.write(root.toString());
            writer.close();

        } catch (IOException e) {
            throw new RuntimeException(
                    String.format("Could not write to the output file at %s", dataset_file.toPath().toString()));
        }
    }

}
