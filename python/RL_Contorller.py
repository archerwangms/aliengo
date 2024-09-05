from WebotsInterface import webot_interface,RL_controller
import torch 
import os 

if __name__ == "__main__":
    Aliengo_sim = webot_interface()
    controller = RL_controller()
    controller.load("./model/model_13000.pt")
    controller.get_inference_policy()
    Aliengo_sim.init_varable()
    print(os.environ["WEBOTS_HOME"])

    while(1):
        controller.step(Aliengo_sim)
        # Aliengo_sim.get_observersion()
        # Aliengo_sim.send_torque(-10.0*Aliengo_sim.joint_vlo)
        # Aliengo_sim.a1_supervisor_class.step(Aliengo_sim.sim_time_step)
        # print(a1_supervisor_node.getDef())

        # zero_torque = torch.zeros(12,dtype=torch.float)
        # Aliengo_sim.send_torque(zero_torque)

